"""MCP helpers and JSON parsing utilities for research and risk."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from servers.manager import MCPManager

logger = logging.getLogger(__name__)


def truncate_text(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [truncated]"


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


def parse_market_clock(result_text: str) -> tuple[bool | None, str]:
    """Parse a get_clock response into (is_open, detail). is_open is None if unparseable."""
    if result_text.startswith("ERROR"):
        return None, truncate_text(result_text, 200)

    raw = parse_mcp_json(result_text)
    if raw is None:
        return None, "could not parse clock response"

    payload = unwrap_alpaca_payload(raw)
    if not isinstance(payload, dict):
        return None, truncate_text(result_text, 200)

    is_open = payload.get("is_open")
    if not isinstance(is_open, bool):
        return None, truncate_text(result_text, 200)

    next_open = payload.get("next_open") or "?"
    next_close = payload.get("next_close") or "?"
    if is_open:
        return True, f"open until {next_close}"
    return False, f"next open {next_open}"


async def is_market_open(manager: MCPManager) -> tuple[bool, str]:
    """Return whether the US equity market is open, plus a short status line."""
    try:
        result = await manager.call_tool("get_clock", {})
        result_text = manager.result_to_text(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_clock failed: %s", exc)
        return False, f"clock check failed ({exc})"

    is_open, detail = parse_market_clock(result_text)
    if is_open is None:
        logger.warning("Could not parse market clock: %s", detail)
        return False, f"clock response unparseable ({detail})"
    if not is_open:
        return False, f"market closed ({detail})"
    return True, detail


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


def format_mcp_summary(name: str, args: dict[str, Any], result_text: str) -> str:
    """One-line human-readable summary of an MCP tool call."""
    if result_text.startswith("ERROR"):
        return f"{name} → {truncate_text(result_text, 200)}"

    raw = parse_mcp_json(result_text)
    payload = unwrap_alpaca_payload(raw) if raw is not None else None

    if name == "get_account_info" and isinstance(payload, dict):
        cash = parse_float_field(payload.get("cash") or payload.get("buying_power"))
        equity = parse_float_field(payload.get("equity") or payload.get("portfolio_value"))
        return f"get_account(cash=${cash:,.2f}, equity=${equity:,.2f})"

    if name == "get_all_positions":
        positions = _iter_positions(payload)
        if not positions:
            return "get_positions(0 positions)"
        detail = ", ".join(f"{symbol} {qty:g}" for symbol, qty in positions)
        return f"get_positions({len(positions)}): {detail}"

    if name == "get_stock_bars":
        symbols = args.get("symbols", "")
        prices = parse_latest_prices(result_text)
        if prices:
            price_str = ", ".join(f"{s} ${p:.2f}" for s, p in list(prices.items())[:6])
            return f"get_stock_bars({symbols}) → {price_str}"
        return f"get_stock_bars({symbols})"

    if name == "get_market_movers":
        syms = extract_mover_symbols(result_text, n=5)
        return f"get_market_movers → {', '.join(syms) or 'none'}"

    if name == "get_clock" and isinstance(payload, dict):
        is_open = payload.get("is_open")
        if is_open is True:
            return f"get_clock(open, closes {payload.get('next_close', '?')})"
        if is_open is False:
            return f"get_clock(closed, opens {payload.get('next_open', '?')})"

    if name == "place_stock_order":
        return f"place_stock_order({args}) → {truncate_text(result_text, 150)}"

    arg_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
    label = f"{name}({arg_str})" if arg_str else name
    return f"{label} → {truncate_text(result_text, 150)}"


def _iter_positions(payload: Any) -> list[tuple[str, float]]:
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
    return [symbol for symbol, _ in _iter_positions(unwrap_alpaca_payload(raw))]


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
