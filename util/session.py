"""US cash-session helpers for weekday cron schedules."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def parse_session_times(raw: str) -> list[tuple[int, int]]:
    """Parse 'HH:MM,HH:MM' into (hour, minute) tuples."""
    times: list[tuple[int, int]] = []
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            continue
        hour_s, minute_s = token.split(":", 1)
        try:
            hour = int(hour_s)
            minute = int(minute_s)
        except ValueError:
            continue
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            times.append((hour, minute))
    return times


def now_in_zone(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def is_weekday(tz_name: str) -> bool:
    return now_in_zone(tz_name).weekday() < 5


def during_regular_hours(tz_name: str, *, start_hour: int = 9, end_hour: int = 16) -> bool:
    """True Mon–Fri between start_hour (inclusive) and end_hour (exclusive) in tz."""
    now = now_in_zone(tz_name)
    if now.weekday() >= 5:
        return False
    return start_hour <= now.hour < end_hour
