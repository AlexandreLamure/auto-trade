"""Equity ticker sanitization for Alpaca calls and the event store."""
# TODO: We could fetch the tradeable symbols from Alpaca directly. It would be slow, so we should fetch once per cycle.

from __future__ import annotations

import re

_EQUITY_RE = re.compile(r"^[A-Z]{1,5}$")
_INDEX_BLOCKLIST = frozenset({"SPX", "VIX", "NDX", "DJIA", "DJI", "COMP"})


def is_tradeable_equity_symbol(symbol: str) -> bool:
    """True for plain US equity tickers Alpaca can fetch (no futures/crypto/indexes)."""
    sym = (symbol or "").upper().strip()
    if not sym or not _EQUITY_RE.fullmatch(sym):
        return False
    if sym in _INDEX_BLOCKLIST:
        return False
    if len(sym) >= 5 and sym.endswith("W"):
        return False
    return True


def sanitize_symbol_list(symbols: list[str] | None) -> list[str]:
    """Deduplicate and drop non-equity symbols, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols or []:
        sym = str(raw).upper().strip()
        if not is_tradeable_equity_symbol(sym) or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out
