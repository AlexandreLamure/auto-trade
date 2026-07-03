"""MCP helpers and JSON parsing utilities for research and risk."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from servers.manager import MCPManager

logger = logging.getLogger(__name__)


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


def extract_mover_symbols(movers_result: str, n: int = 3) -> list[str]:
    """Return up to *n* ticker symbols from a get_market_movers tool result."""
    raw = parse_mcp_json(movers_result)
    if raw is None:
        return []

    payload = unwrap_alpaca_payload(raw)
    if not isinstance(payload, dict):
        return []
    gainers = payload.get("gainers") or []
    if not isinstance(gainers, list):
        return []

    ranked: list[tuple[float, str]] = []
    for item in gainers:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        price = float(item.get("price") or 0)
        pct = abs(float(item.get("percent_change") or 0))
        penalty = 0.0
        if symbol.endswith("W"):
            penalty += 1000.0
        if price < 1.0:
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
        if len(symbols) >= n:
            break
    return symbols


async def call_mcp_tool(
    manager: MCPManager,
    name: str,
    arguments: dict[str, Any] | None = None,
    tool_call_log: list[dict[str, Any]] | None = None,
    *,
    result_truncate: int = 8000,
) -> str:
    """Execute one MCP tool and optionally append to *tool_call_log*."""
    args = arguments or {}
    logger.info("MCP tool: %s(%s)", name, args)
    try:
        result = await manager.call_tool(name, args)
        result_text = manager.result_to_text(result)
    except Exception as exc:  # noqa: BLE001
        result_text = f"ERROR executing tool '{name}': {exc}"
        logger.warning(result_text)

    if tool_call_log is not None:
        tool_call_log.append(
            {"name": name, "args": args, "result": result_text[:result_truncate]}
        )
    return result_text


def extract_position_symbols(positions_result: str) -> list[str]:
    """Return held ticker symbols from get_all_positions JSON."""
    raw = parse_mcp_json(positions_result)
    if raw is None:
        return []

    payload = unwrap_alpaca_payload(raw)
    if isinstance(payload, list):
        positions = payload
    elif isinstance(payload, dict):
        positions = payload.get("result") or payload.get("positions") or []
    else:
        return []

    if not isinstance(positions, list):
        return []

    symbols: list[str] = []
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        symbol = str(pos.get("symbol", "")).upper().strip()
        if symbol:
            symbols.append(symbol)
    return symbols


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


def parse_latest_prices(bars_json: str) -> dict[str, float]:
    """Extract the latest daily close price per symbol from get_stock_bars JSON."""
    raw = parse_mcp_json(bars_json)
    if raw is None:
        return {}

    payload = unwrap_alpaca_payload(raw)
    if not isinstance(payload, dict):
        return {}

    bars_by_symbol = payload.get("bars") or {}
    if not isinstance(bars_by_symbol, dict):
        return {}

    prices: dict[str, float] = {}
    for symbol, bars in bars_by_symbol.items():
        if not isinstance(bars, list) or not bars:
            continue
        last_bar = bars[-1]
        if not isinstance(last_bar, dict):
            continue
        close = parse_float_field(last_bar.get("c") or last_bar.get("close"))
        if close > 0:
            prices[str(symbol).upper()] = close
    return prices
