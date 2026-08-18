"""Alpaca MCP helpers shared by the news watchlist and trading research."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from servers.manager import MCPManager

logger = logging.getLogger(__name__)


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


def iter_positions(payload: Any) -> list[tuple[str, float]]:
    """Return (symbol, quantity) pairs from an Alpaca positions payload."""
    if isinstance(payload, list):
        positions = payload
    elif isinstance(payload, dict):
        positions = payload.get("result") or payload.get("positions") or []
    else:
        return []
    if not isinstance(positions, list):
        return []

    held: list[tuple[str, float]] = []
    for pos in positions:
        if not isinstance(pos, dict):
            continue
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
