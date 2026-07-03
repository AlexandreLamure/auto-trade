"""
Pre-trade risk validation for committee portfolio decisions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from agent.decision import TradeOrder
from agent.workflow import parse_float_field, parse_mcp_json, unwrap_alpaca_payload
from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationContext:
    cash_available: float
    held_quantities: dict[str, float]
    latest_prices: dict[str, float]
    max_position_pct: float
    max_orders: int


def parse_held_quantities(positions_json: str) -> dict[str, float]:
    raw = parse_mcp_json(positions_json)
    if raw is None:
        return {}
    payload = unwrap_alpaca_payload(raw)
    if isinstance(payload, list):
        positions = payload
    elif isinstance(payload, dict):
        positions = payload.get("result") or payload.get("positions") or []
    else:
        return {}
    if not isinstance(positions, list):
        return {}
    held: dict[str, float] = {}
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        symbol = str(pos.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        qty = parse_float_field(pos.get("qty") or pos.get("quantity"))
        held[symbol] = qty
    return held


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


def validate_orders(
    orders: list[TradeOrder],
    ctx: ValidationContext,
) -> tuple[list[TradeOrder], list[str]]:
    """Return (approved_orders, rejection_reasons)."""
    rejections: list[str] = []
    netted = net_orders(orders)

    if len(netted) > ctx.max_orders:
        rejections.append(
            f"Too many orders ({len(netted)}); max is {ctx.max_orders}. Keeping first {ctx.max_orders}."
        )
        netted = netted[: ctx.max_orders]

    approved: list[TradeOrder] = []
    cash_remaining = ctx.cash_available
    max_per_buy = ctx.cash_available * ctx.max_position_pct
    if max_per_buy <= 0 and any(o.side == "buy" for o in netted):
        logger.warning(
            "Buy validation blocked: cash_available=%.2f max_position_pct=%.2f",
            ctx.cash_available,
            ctx.max_position_pct,
        )
    sell_reserved: dict[str, float] = {}

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
            approved.append(
                TradeOrder(
                    symbol=order.symbol,
                    side="sell",
                    quantity=qty,
                    rationale=order.rationale,
                )
            )
        else:
            if max_per_buy <= 0:
                rejections.append(f"BUY {order.symbol}: max position size is zero")
                continue
            if cash_remaining <= 0:
                rejections.append(f"BUY {order.symbol}: insufficient cash")
                continue

            price = ctx.latest_prices.get(order.symbol)
            qty = order.quantity
            if price and price > 0:
                max_shares = math.floor(max_per_buy / price)
                max_affordable = math.floor(cash_remaining / price)
                capped = min(qty, max_shares, max_affordable)
                if capped < qty:
                    rejections.append(
                        f"BUY {order.symbol}: reduced qty {qty:.0f} → {capped:.0f} "
                        f"(max {ctx.max_position_pct:.0%} of cash @ ${price:.2f})"
                    )
                qty = capped
            else:
                rejections.append(
                    f"BUY {order.symbol}: no price data; cannot size order safely"
                )
                continue

            if qty < 1:
                rejections.append(
                    f"BUY {order.symbol}: qty below 1 after cash cap "
                    f"(price=${price:.2f}, max ${max_per_buy:.2f} per buy)"
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
) -> ValidationContext:
    return ValidationContext(
        cash_available=cash_available,
        held_quantities=parse_held_quantities(positions_json),
        latest_prices=latest_prices or {},
        max_position_pct=settings.max_position_pct,
        max_orders=settings.max_orders_per_cycle,
    )
