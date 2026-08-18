"""US cash-session helpers (America/New_York)."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


def now_et() -> datetime:
    return datetime.now(MARKET_TZ)


def is_market_open() -> bool:
    """True Mon–Fri during the US cash session (09:30–16:00 ET)."""
    now = now_et()
    if now.weekday() >= 5:
        return False
    return SESSION_OPEN <= now.time() < SESSION_CLOSE


def market_closed_message() -> str:
    now = now_et()
    return f"SKIP – market closed ({now.strftime('%a %Y-%m-%d %H:%M %Z')})"
