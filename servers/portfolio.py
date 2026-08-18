"""Alpaca MCP helpers shared by the news watchlist and trading research."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from servers.manager import MCPManager

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0
    cost_basis: float = 0.0
    change_today: float = 0.0
    days_held: int | None = None


@dataclass
class PositionsAndMovers:
    positions_json: str
    movers_json: str
    held: list[str]
    mover_symbols: list[str]


def parse_mcp_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def unwrap_alpaca_payload(raw: Any) -> Any:
    """Unwrap Alpaca MCP responses: {\"_alpaca_mcp_security\": ..., \"data\": ...}."""
    if not isinstance(raw, dict):
        return raw
    if "data" in raw:
        return raw["data"]
    return raw


def parse_float_field(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _position_dicts(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        positions = payload
    elif isinstance(payload, dict):
        positions = payload.get("result") or payload.get("positions") or []
    else:
        return []
    if not isinstance(positions, list):
        return []
    return [pos for pos in positions if isinstance(pos, dict)]


def parse_positions(positions_json: str) -> list[Position]:
    """Parse Alpaca positions into structured holdings with P&L fields."""
    raw = parse_mcp_json(positions_json)
    if raw is None:
        return []
    result: list[Position] = []
    for pos in _position_dicts(unwrap_alpaca_payload(raw)):
        symbol = str(pos.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        qty = parse_float_field(pos.get("qty") or pos.get("quantity"))
        avg = parse_float_field(pos.get("avg_entry_price") or pos.get("avg_entry"))
        price = parse_float_field(pos.get("current_price") or pos.get("asset_current_price"))
        market_value = parse_float_field(pos.get("market_value"))
        if market_value == 0.0 and price > 0 and qty:
            market_value = price * qty
        cost = parse_float_field(pos.get("cost_basis") or pos.get("cost"))
        if cost == 0.0 and avg > 0 and qty:
            cost = avg * abs(qty)
        unrealized = parse_float_field(pos.get("unrealized_pl") or pos.get("unrealized_intraday_pl"))
        plpc = parse_float_field(pos.get("unrealized_plpc"))
        if plpc == 0.0 and cost:
            plpc = unrealized / abs(cost)
        result.append(
            Position(
                symbol=symbol,
                qty=qty,
                avg_entry_price=avg,
                current_price=price,
                market_value=market_value,
                unrealized_pl=unrealized,
                unrealized_plpc=plpc,
                cost_basis=cost,
                change_today=parse_float_field(
                    pos.get("change_today") or pos.get("unrealized_intraday_pl")
                ),
            )
        )
    return result


def estimate_days_held(orders_json: str) -> dict[str, int]:
    """Approximate days held from the oldest filled buy in recent orders."""
    from datetime import datetime, timezone

    from util.time import parse_iso, utcnow

    raw = parse_mcp_json(orders_json)
    if raw is None:
        return {}
    payload = unwrap_alpaca_payload(raw)
    orders = payload if isinstance(payload, list) else []
    if isinstance(payload, dict):
        orders = payload.get("orders") or payload.get("result") or []
    if not isinstance(orders, list):
        return {}

    oldest: dict[str, datetime] = {}
    now = utcnow()
    for item in orders:
        if not isinstance(item, dict):
            continue
        side = str(item.get("side", "")).lower()
        status = str(item.get("status", "")).lower()
        if side != "buy" or status not in ("filled", "partially_filled"):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        filled_at = parse_iso(
            str(item.get("filled_at") or item.get("submitted_at") or "")
        )
        if not symbol or filled_at is None:
            continue
        if filled_at.tzinfo is None:
            filled_at = filled_at.replace(tzinfo=timezone.utc)
        previous = oldest.get(symbol)
        if previous is None or filled_at < previous:
            oldest[symbol] = filled_at

    days: dict[str, int] = {}
    for symbol, when in oldest.items():
        days[symbol] = max(0, (now - when).days)
    return days


def format_holdings_table(
    positions: list[Position], *, equity: float
) -> str:
    """Markdown table: qty, cost, last, unrealized, weight, days held."""
    if not positions:
        return "_No open positions._"
    lines = [
        "| Symbol | Qty | Avg cost | Last | Unrealized | Weight | Days held |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pos in positions:
        weight = (pos.market_value / equity * 100.0) if equity > 0 else 0.0
        pl_pct = pos.unrealized_plpc * 100.0
        # Alpaca sometimes already stores percent (e.g. 1.5 meaning 150%).
        if abs(pos.unrealized_plpc) > 2:
            pl_pct = pos.unrealized_plpc
        days = "—" if pos.days_held is None else str(pos.days_held)
        lines.append(
            f"| {pos.symbol} | {pos.qty:g} | ${pos.avg_entry_price:.2f} | "
            f"${pos.current_price:.2f} | {pl_pct:+.1f}% (${pos.unrealized_pl:,.0f}) | "
            f"{weight:.1f}% | {days} |"
        )
    return "\n".join(lines)


def iter_positions(payload: Any) -> list[tuple[str, float]]:
    """Return (symbol, quantity) pairs from an Alpaca positions payload."""
    held: list[tuple[str, float]] = []
    for pos in _position_dicts(payload):
        symbol = str(pos.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        qty = parse_float_field(pos.get("qty") or pos.get("quantity"))
        held.append((symbol, qty))
    return held


def extract_position_symbols(positions_result: str) -> list[str]:
    """Return held ticker symbols from get_all_positions JSON."""
    raw = parse_mcp_json(positions_result)
    if raw is None:
        return []
    return [symbol for symbol, _ in iter_positions(unwrap_alpaca_payload(raw))]


def _rank_mover_side(items: Any, *, min_price: float, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    ranked: list[tuple[float, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        price = parse_float_field(item.get("price") or item.get("p"))
        pct = abs(
            parse_float_field(
                item.get("percent_change")
                or item.get("change_percent")
                or item.get("percent")
            )
        )
        penalty = 0.0
        if symbol.endswith("W"):
            penalty += 1000.0
        if 0 < price < min_price:
            penalty += 500.0
        ranked.append((penalty - pct, symbol))
    ranked.sort(key=lambda pair: pair[0])
    symbols: list[str] = []
    seen: set[str] = set()
    for _, symbol in ranked:
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def extract_mover_symbols(
    movers_result: str,
    n: int = 3,
    *,
    n_gainers: int | None = None,
    n_losers: int | None = None,
    min_price: float = 5.0,
) -> list[str]:
    """Return gainer and loser tickers from get_market_movers JSON."""
    raw = parse_mcp_json(movers_result)
    if raw is None:
        return []

    payload = unwrap_alpaca_payload(raw)
    if not isinstance(payload, dict):
        return []

    gainer_n = n_gainers if n_gainers is not None else max(1, (n + 1) // 2)
    loser_n = n_losers if n_losers is not None else max(1, n // 2)
    gainers = _rank_mover_side(
        payload.get("gainers") or [], min_price=min_price, limit=gainer_n
    )
    losers = _rank_mover_side(
        payload.get("losers") or [], min_price=min_price, limit=loser_n
    )
    merged: list[str] = []
    seen: set[str] = set()
    # Interleave so a gainer-heavy tape cannot crowd out oversold names.
    for pair in zip(gainers, losers):
        for symbol in pair:
            if symbol in seen:
                continue
            seen.add(symbol)
            merged.append(symbol)
    for symbol in gainers[len(losers) :] + losers[len(gainers) :]:
        if symbol in seen:
            continue
        seen.add(symbol)
        merged.append(symbol)
    return merged


def merge_symbol_universe(
    held: list[str],
    movers: list[str],
    *,
    max_candidates: int,
) -> tuple[list[str], list[str]]:
    """Return (held_symbols, candidate_symbols) deduplicated."""
    held_unique = list(dict.fromkeys(s.upper() for s in held))
    candidates: list[str] = []
    seen = set(held_unique)
    for symbol in movers:
        sym = symbol.upper()
        if sym in seen:
            continue
        seen.add(sym)
        candidates.append(sym)
        if len(candidates) >= max_candidates:
            break
    return held_unique, candidates


async def call_mcp_tool(
    manager: MCPManager,
    name: str,
    arguments: dict[str, Any] | None = None,
    tool_call_log: list[dict[str, Any]] | None = None,
) -> str:
    """Execute one MCP tool and optionally append to *tool_call_log*."""
    args = arguments or {}
    try:
        result = await manager.call_tool(name, args)
        result_text = manager.result_to_text(result)
    except Exception as exc:  # noqa: BLE001
        result_text = f"ERROR executing tool '{name}': {exc}"
        logger.warning(result_text)

    if tool_call_log is not None:
        tool_call_log.append({"name": name, "args": args, "result": result_text})
    return result_text


async def load_positions_and_movers(
    manager: MCPManager,
    *,
    n_movers: int,
    min_price: float = 5.0,
    tool_call_log: list[dict[str, Any]] | None = None,
) -> PositionsAndMovers:
    """Fetch open positions and market movers in one place."""
    positions_json = await call_mcp_tool(
        manager, "get_all_positions", tool_call_log=tool_call_log
    )
    movers_json = await call_mcp_tool(
        manager,
        "get_market_movers",
        {"market_type": "stocks"},
        tool_call_log=tool_call_log,
    )
    return PositionsAndMovers(
        positions_json=positions_json,
        movers_json=movers_json,
        held=extract_position_symbols(positions_json),
        mover_symbols=extract_mover_symbols(
            movers_json, n=n_movers, min_price=min_price
        ),
    )
