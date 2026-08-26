"""MCP helpers and JSON parsing utilities for research and risk."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from servers.portfolio import (
    extract_mover_symbols,
    iter_positions,
    parse_float_field,
    parse_mcp_json,
    unwrap_alpaca_payload,
)
from util.symbols import is_tradeable_equity_symbol, sanitize_symbol_list
from util.text import truncate_text


@dataclass
class ExecutionRecord:
    symbol: str
    side: str
    submitted_qty: float
    status: str
    limit_price: float | None = None
    reason: str = ""


@dataclass
class PriceStats:
    symbol: str
    price: float
    ret_1d: float
    ret_5d: float
    ret_20d: float
    dist_high: float
    vol_ratio: float
    atr: float
    realized_vol: float
    adv_shares: float
    dollar_adv: float


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
        positions = iter_positions(payload)
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


def parse_nbbo(quote_json: str, symbol: str) -> tuple[float, float]:
    """Return (bid, ask) from get_stock_latest_quote JSON."""
    if quote_json.startswith("ERROR"):
        return 0.0, 0.0
    raw = parse_mcp_json(quote_json)
    if raw is None:
        return 0.0, 0.0
    payload = unwrap_alpaca_payload(raw)
    bid, ask = _extract_bid_ask(payload, symbol.upper())
    if bid > 0 or ask > 0:
        return bid, ask
    return _extract_bid_ask(payload, "")


def _extract_bid_ask(obj: Any, symbol: str, *, depth: int = 0) -> tuple[float, float]:
    if depth > 6 or obj is None:
        return 0.0, 0.0
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0, 0.0
        return _extract_bid_ask(obj, symbol, depth=depth + 1)
    if isinstance(obj, list):
        for item in obj:
            bid, ask = _extract_bid_ask(item, symbol, depth=depth + 1)
            if bid > 0 or ask > 0:
                return bid, ask
        return 0.0, 0.0
    if not isinstance(obj, dict):
        return 0.0, 0.0
    if symbol:
        nested = obj.get(symbol) or obj.get(symbol.lower())
        if nested is not None:
            bid, ask = _extract_bid_ask(nested, "", depth=depth + 1)
            if bid > 0 or ask > 0:
                return bid, ask
    bid = parse_float_field(
        obj.get("bp")
        or obj.get("bid_price")
        or obj.get("bid")
        or obj.get("BidPrice")
        or obj.get("bidPrice")
    )
    ask = parse_float_field(
        obj.get("ap")
        or obj.get("ask_price")
        or obj.get("ask")
        or obj.get("AskPrice")
        or obj.get("askPrice")
    )
    if bid > 0 or ask > 0:
        return bid, ask
    for key in ("quote", "quotes", "latest_quote", "latestQuote"):
        if key in obj:
            bid, ask = _extract_bid_ask(obj[key], symbol, depth=depth + 1)
            if bid > 0 or ask > 0:
                return bid, ask
    return 0.0, 0.0


def parse_fractionable(asset_json: str) -> bool:
    if asset_json.startswith("ERROR"):
        return False
    raw = parse_mcp_json(asset_json)
    if raw is None:
        return False
    payload = unwrap_alpaca_payload(raw)
    flag = _find_key(payload, "fractionable")
    if isinstance(flag, bool):
        return flag
    if flag is None:
        return False
    return str(flag).lower() in {"true", "1", "yes"}


def parse_asset_name(asset_json: str) -> str:
    if asset_json.startswith("ERROR"):
        return ""
    raw = parse_mcp_json(asset_json)
    if raw is None:
        return ""
    payload = unwrap_alpaca_payload(raw)
    name = _find_key(payload, "name") or _find_key(payload, "official_name")
    return str(name).strip() if name else ""


def parse_order_status(order_json: str) -> str:
    """Best-effort Alpaca order status from place_stock_order JSON."""
    if order_json.startswith("ERROR"):
        return "failed"
    raw = parse_mcp_json(order_json)
    if raw is None:
        return "submitted"
    payload = unwrap_alpaca_payload(raw)
    status = _find_key(payload, "status")
    if not status:
        return "submitted"
    normalized = str(status).lower().strip()
    if normalized in {"filled", "partially_filled"}:
        return "filled" if normalized == "filled" else "submitted"
    if normalized in {"rejected", "canceled", "cancelled", "expired"}:
        return normalized.replace("cancelled", "canceled")
    if normalized in {"accepted", "new", "pending_new", "held", "done_for_day"}:
        return "submitted"
    return "submitted"


def _find_key(obj: Any, key: str, *, depth: int = 0) -> Any:
    if depth > 6 or obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return _find_key(obj, key, depth=depth + 1)
    if isinstance(obj, list):
        for item in obj:
            found = _find_key(item, key, depth=depth + 1)
            if found is not None:
                return found
        return None
    if not isinstance(obj, dict):
        return None
    if key in obj:
        return obj[key]
    for value in obj.values():
        if isinstance(value, (dict, list, str)):
            found = _find_key(value, key, depth=depth + 1)
            if found is not None:
                return found
    return None


def format_order_qty(qty: float, *, fractionable: bool) -> str | None:
    """Format a risk-approved qty for Alpaca. Never rounds up past *qty*."""
    if qty <= 0:
        return None
    if not fractionable:
        whole = int(qty)
        if whole < 1:
            return None
        return str(whole)
    if abs(qty - round(qty)) < 1e-8:
        whole = int(round(qty))
        return str(whole) if whole >= 1 else None
    formatted = f"{qty:.6f}".rstrip("0").rstrip(".")
    return formatted if formatted and float(formatted) > 0 else None


def _parse_bars_by_symbol(bars_json: str) -> dict[str, list[dict[str, Any]]]:
    """Return OHLCV bar lists keyed by symbol from get_stock_bars JSON."""
    raw = parse_mcp_json(bars_json)
    if raw is None:
        return {}
    payload = unwrap_alpaca_payload(raw)
    if not isinstance(payload, dict):
        return {}
    bars_by_symbol = payload.get("bars") or {}
    if not isinstance(bars_by_symbol, dict):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for symbol, bars in bars_by_symbol.items():
        if isinstance(bars, list):
            result[str(symbol).upper()] = [b for b in bars if isinstance(b, dict)]
    return result


def merge_bars_json(parts: list[str]) -> str:
    """Combine multiple get_stock_bars responses into one bars payload."""
    merged: dict[str, list[dict[str, Any]]] = {}
    for text in parts:
        if not text or text.startswith("ERROR"):
            continue
        merged.update(_parse_bars_by_symbol(text))
    return json.dumps({"bars": merged})


def compute_price_stats(
    bars_json: str, symbols: list[str] | None = None
) -> dict[str, PriceStats]:
    """Per-symbol returns, ATR, realized vol, and average volume from daily bars."""
    bars_by_symbol = _parse_bars_by_symbol(bars_json)
    if symbols is not None:
        allow = {s.upper() for s in symbols}
        bars_by_symbol = {
            key: value for key, value in bars_by_symbol.items() if key in allow
        }
    stats: dict[str, PriceStats] = {}
    for symbol, bars in bars_by_symbol.items():
        if len(bars) < 2:
            continue
        closes = [parse_float_field(b.get("c") or b.get("close")) for b in bars]
        highs = [parse_float_field(b.get("h") or b.get("high"), closes[i]) for i, b in enumerate(bars)]
        lows = [parse_float_field(b.get("l") or b.get("low"), closes[i]) for i, b in enumerate(bars)]
        volumes = [parse_float_field(b.get("v") or b.get("volume")) for b in bars]
        price = closes[-1]
        if price <= 0:
            continue
        ret_1d = ((closes[-1] / closes[-2]) - 1.0) * 100.0 if len(closes) >= 2 and closes[-2] else 0.0
        ret_5d = ((closes[-1] / closes[-6]) - 1.0) * 100.0 if len(closes) >= 6 and closes[-6] else 0.0
        ret_20d = ((closes[-1] / closes[-21]) - 1.0) * 100.0 if len(closes) >= 21 and closes[-21] else 0.0
        high_30 = max(closes)
        dist_high = ((price / high_30) - 1.0) * 100.0 if high_30 > 0 else 0.0
        recent_vols = volumes[-20:] or volumes
        vol_avg = sum(recent_vols) / len(recent_vols) if recent_vols else 0.0
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0
        true_ranges: list[float] = []
        for i in range(1, len(bars)):
            prev_close = closes[i - 1]
            tr = max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
            true_ranges.append(tr)
        atr_window = true_ranges[-14:] or true_ranges
        atr = sum(atr_window) / len(atr_window) if atr_window else 0.0
        rets: list[float] = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                rets.append(closes[i] / closes[i - 1] - 1.0)
        sample = rets[-20:] or rets
        if len(sample) >= 2:
            avg = sum(sample) / len(sample)
            var = sum((r - avg) ** 2 for r in sample) / (len(sample) - 1)
            realized_vol = math.sqrt(var) * math.sqrt(252.0)
        else:
            realized_vol = 0.0
        dollar_adv = 0.0
        if recent_vols:
            recent_closes = closes[-len(recent_vols) :]
            dollar_adv = sum(c * v for c, v in zip(recent_closes, recent_vols)) / len(recent_vols)
        stats[symbol] = PriceStats(
            symbol=symbol,
            price=price,
            ret_1d=ret_1d,
            ret_5d=ret_5d,
            ret_20d=ret_20d,
            dist_high=dist_high,
            vol_ratio=vol_ratio,
            atr=atr,
            realized_vol=realized_vol,
            adv_shares=vol_avg,
            dollar_adv=dollar_adv,
        )
    return stats


def compute_price_analytics(
    bars_json: str,
    symbols: list[str] | None = None,
    *,
    earnings: dict[str, str] | None = None,
) -> str:
    """Build a compact markdown table from bar data."""
    stats = compute_price_stats(bars_json, symbols)
    if not stats:
        return "_No price analytics available._"

    lines = [
        "| Symbol | Price | 1d% | 5d% | 20d% | ATR | Vol (ann) | ADV $ | vs 30d High | Earn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for symbol in sorted(stats):
        row = stats[symbol]
        earn = (earnings or {}).get(symbol, "—")
        lines.append(
            f"| {symbol} | ${row.price:.2f} | {row.ret_1d:+.1f}% | {row.ret_5d:+.1f}% | "
            f"{row.ret_20d:+.1f}% | ${row.atr:.2f} | {row.realized_vol * 100:.0f}% | "
            f"${row.dollar_adv/1e6:.1f}M | {row.dist_high:+.1f}% | {earn} |"
        )
    return "\n".join(lines)


def expand_symbol_universe(
    held: list[str],
    movers: list[str],
    event_tickers: list[str],
    *,
    max_candidates: int,
    mover_slots: int | None = None,
    event_slots: int | None = None,
) -> tuple[list[str], list[str]]:
    """Merge held, movers, and event-discovered tickers into a research universe.

    Event tickers get reserved slots first so micro-cap movers cannot crowd them out.
    Remaining capacity is filled from leftover events, then leftover movers.
    """
    held_unique = sanitize_symbol_list(held)
    mover_cap = mover_slots if mover_slots is not None else max_candidates
    event_cap = event_slots if event_slots is not None else max_candidates
    seen = set(held_unique)
    candidates: list[str] = []

    def _take(source: list[str], limit: int) -> None:
        added = 0
        for symbol in source:
            if added >= limit or len(candidates) >= max_candidates:
                return
            if not is_tradeable_equity_symbol(symbol):
                continue
            sym = symbol.upper().strip()
            if sym in seen:
                continue
            seen.add(sym)
            candidates.append(sym)
            added += 1

    _take(event_tickers, event_cap)
    _take(movers, mover_cap)
    _take(event_tickers, max_candidates)
    _take(movers, max_candidates)
    return held_unique, candidates


def filter_liquid_candidates(
    candidates: list[str],
    bars_json: str,
    *,
    min_price: float,
    min_adv_shares: float,
) -> list[str]:
    """Drop warrants, sub-min-price names, and thin volume from candidates."""
    bars_by_symbol = _parse_bars_by_symbol(bars_json)
    kept: list[str] = []
    for symbol in candidates:
        if not is_tradeable_equity_symbol(symbol):
            continue
        sym = symbol.upper()
        bars = bars_by_symbol.get(sym) or []
        if not bars:
            continue
        closes = [parse_float_field(b.get("c") or b.get("close")) for b in bars]
        volumes = [parse_float_field(b.get("v") or b.get("volume")) for b in bars]
        price = closes[-1] if closes else 0.0
        if price < min_price:
            continue
        recent = volumes[-20:] or volumes
        adv = sum(recent) / len(recent) if recent else 0.0
        if adv < min_adv_shares:
            continue
        kept.append(sym)
    return kept
