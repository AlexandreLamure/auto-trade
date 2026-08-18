"""
Pre-trade risk validation for committee portfolio decisions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from agent.decision import TradeOrder
from config.settings import Settings
from servers.portfolio import Position, parse_positions
from agent.workflow import PriceStats

logger = logging.getLogger(__name__)


@dataclass
class ValidationContext:
    cash_available: float
    portfolio_equity: float
    held_quantities: dict[str, float]
    held_market_value: dict[str, float]
    latest_prices: dict[str, float]
    max_position_pct: float
    max_orders: int
    price_stats: dict[str, PriceStats]
    earnings_flags: dict[str, str]
    risk_per_name_pct: float
    stop_atr_multiple: float
    stop_pct: float
    block_earnings_buys: bool


def stop_distance(
    price: float,
    stats: PriceStats | None,
    *,
    stop_atr_multiple: float,
    stop_pct: float,
) -> float:
    """Dollar distance from price to stop, using ATR with a percent floor."""
    atr_stop = (stats.atr * stop_atr_multiple) if stats and stats.atr > 0 else 0.0
    pct_stop = price * stop_pct
    return max(atr_stop, pct_stop, price * 0.02)


def inject_time_stops(
    orders: list[TradeOrder],
    holdings: list[Position],
    *,
    time_stop_days: int,
) -> list[TradeOrder]:
    """Add full sells for held losers at or past the horizon."""
    if time_stop_days <= 0:
        return orders
    already = {(o.symbol, o.side) for o in orders}
    extra: list[TradeOrder] = []
    for pos in holdings:
        if pos.days_held is None or pos.days_held < time_stop_days:
            continue
        plpc = pos.unrealized_plpc
        if abs(plpc) > 2:
            plpc = plpc / 100.0
        if plpc >= 0:
            continue
        if (pos.symbol, "sell") in already:
            continue
        extra.append(
            TradeOrder(
                symbol=pos.symbol,
                side="sell",
                quantity=pos.qty,
                rationale=f"time-stop: held {pos.days_held}d with {plpc:.1%} PnL",
            )
        )
    return extra + orders


def parse_held_quantities(positions_json: str) -> dict[str, float]:
    return {pos.symbol: pos.qty for pos in parse_positions(positions_json)}


def net_orders(orders: list[TradeOrder]) -> list[TradeOrder]:
    """Net buy/sell on the same symbol into a single order per side."""
    nets: dict[tuple[str, str], float] = {}
    rationales: dict[tuple[str, str], str] = {}
    for order in orders:
        key = (order.symbol, order.side)
        nets[key] = nets.get(key, 0.0) + order.quantity
        if order.rationale:
            rationales[key] = order.rationale
    result: list[TradeOrder] = []
    for (symbol, side), qty in nets.items():
        if qty <= 0:
            continue
        result.append(
            TradeOrder(
                symbol=symbol,
                side=side,  # type: ignore[arg-type]
                quantity=qty,
                rationale=rationales.get((symbol, side), ""),
            )
        )
    return result


def _sells_first(orders: list[TradeOrder]) -> list[TradeOrder]:
    sells = [o for o in orders if o.side == "sell"]
    buys = [o for o in orders if o.side != "sell"]
    return sells + buys


def validate_orders(
    orders: list[TradeOrder],
    ctx: ValidationContext,
) -> tuple[list[TradeOrder], list[str]]:
    """Return (approved_orders, rejection_reasons). Sells run before buys."""
    rejections: list[str] = []
    netted = _sells_first(net_orders(orders))

    if len(netted) > ctx.max_orders:
        rejections.append(
            f"Too many orders ({len(netted)}); max is {ctx.max_orders}. "
            "Keeping sells first, then remaining buys."
        )
        netted = netted[: ctx.max_orders]

    approved: list[TradeOrder] = []
    cash_remaining = ctx.cash_available
    equity = ctx.portfolio_equity if ctx.portfolio_equity > 0 else ctx.cash_available
    max_name_notional = equity * ctx.max_position_pct
    if max_name_notional <= 0 and any(o.side == "buy" for o in netted):
        logger.warning(
            "Buy validation blocked: equity=%.2f max_position_pct=%.2f",
            equity,
            ctx.max_position_pct,
        )
    sell_reserved: dict[str, float] = {}
    sold_notional: dict[str, float] = {}

    for order in netted:
        if order.side == "sell":
            held = ctx.held_quantities.get(order.symbol, 0.0)
            already = sell_reserved.get(order.symbol, 0.0)
            allowed = max(0.0, held - already)
            if allowed <= 0:
                rejections.append(f"SELL {order.symbol}: no position to sell")
                continue
            qty = min(order.quantity, allowed)
            sell_reserved[order.symbol] = already + qty
            price = ctx.latest_prices.get(order.symbol) or 0.0
            proceeds = qty * price if price > 0 else 0.0
            cash_remaining += proceeds
            sold_notional[order.symbol] = sold_notional.get(order.symbol, 0.0) + proceeds
            approved.append(
                TradeOrder(
                    symbol=order.symbol,
                    side="sell",
                    quantity=qty,
                    rationale=order.rationale,
                )
            )
            continue

        if max_name_notional <= 0:
            rejections.append(f"BUY {order.symbol}: max position size is zero")
            continue
        if cash_remaining <= 0:
            rejections.append(f"BUY {order.symbol}: insufficient cash")
            continue

        if ctx.block_earnings_buys and order.symbol in ctx.earnings_flags:
            rejections.append(
                f"BUY {order.symbol}: earnings window ({ctx.earnings_flags[order.symbol]})"
            )
            continue

        price = ctx.latest_prices.get(order.symbol)
        qty = order.quantity
        if not price or price <= 0:
            rejections.append(
                f"BUY {order.symbol}: no price data; cannot size order safely"
            )
            continue

        existing_mv = ctx.held_market_value.get(order.symbol, 0.0)
        existing_mv -= sold_notional.get(order.symbol, 0.0)
        room = max(0.0, max_name_notional - max(0.0, existing_mv))
        stats = ctx.price_stats.get(order.symbol)
        stop_dist = stop_distance(
            price,
            stats,
            stop_atr_multiple=ctx.stop_atr_multiple,
            stop_pct=ctx.stop_pct,
        )
        risk_budget = equity * ctx.risk_per_name_pct if ctx.risk_per_name_pct > 0 else room
        vol_shares = (
            math.floor(risk_budget / stop_dist * 10000) / 10000 if stop_dist > 0 else qty
        )
        max_shares = math.floor(room / price * 10000) / 10000 if room > 0 else 0.0
        max_affordable = math.floor(cash_remaining / price * 10000) / 10000
        capped = min(qty, max_shares, max_affordable, vol_shares)
        if capped < qty:
            rejections.append(
                f"BUY {order.symbol}: reduced qty {qty:g} → {capped:g} "
                f"(equity cap / vol-aware stop ${stop_dist:.2f})"
            )
        qty = capped

        if qty <= 0:
            rejections.append(
                f"BUY {order.symbol}: qty below minimum after equity/cash cap "
                f"(price=${price:.2f}, max ${max_name_notional:.2f} per name)"
            )
            continue

        cost = qty * price
        if cost > cash_remaining:
            rejections.append(f"BUY {order.symbol}: cost ${cost:.2f} exceeds cash")
            continue

        approved.append(
            TradeOrder(
                symbol=order.symbol,
                side="buy",
                quantity=qty,
                rationale=order.rationale,
            )
        )
        cash_remaining -= cost

    return approved, rejections


def build_validation_context(
    settings: Settings,
    *,
    cash_available: float,
    positions_json: str,
    latest_prices: dict[str, float] | None = None,
    portfolio_equity: float = 0.0,
    price_stats: dict[str, PriceStats] | None = None,
    earnings_flags: dict[str, str] | None = None,
) -> ValidationContext:
    prices = latest_prices or {}
    holdings = parse_positions(positions_json)
    held_qty = {pos.symbol: pos.qty for pos in holdings}
    held_mv: dict[str, float] = {}
    for pos in holdings:
        value = pos.market_value
        if value <= 0:
            px = pos.current_price or prices.get(pos.symbol, 0.0)
            value = abs(pos.qty) * px
        held_mv[pos.symbol] = value
    return ValidationContext(
        cash_available=cash_available,
        portfolio_equity=portfolio_equity,
        held_quantities=held_qty or parse_held_quantities(positions_json),
        held_market_value=held_mv,
        latest_prices=prices,
        max_position_pct=settings.max_position_pct,
        max_orders=settings.max_orders_per_cycle,
        price_stats=price_stats or {},
        earnings_flags=earnings_flags or {},
        risk_per_name_pct=settings.risk_per_name_pct,
        stop_atr_multiple=settings.stop_atr_multiple,
        stop_pct=settings.stop_pct,
        block_earnings_buys=settings.block_earnings_buys,
    )
