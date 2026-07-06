"""Shared helpers for HTTP API news sources."""

from __future__ import annotations

from news.models import Signal
from news.sources.base import extract_tickers


def valid_api_key(key: str) -> bool:
    return bool(key) and not key.startswith("your_")


def api_item_to_signal(
    *,
    url: str,
    title: str,
    snippet: str,
    published,
    source_label: str,
    watchlist: list[str],
    tickers_hint: list[str] | None = None,
) -> Signal | None:
    if not url:
        return None
    title = title.strip()
    snippet = (snippet or title)[:500]
    hints = list(tickers_hint or [])
    extracted = extract_tickers(f"{title} {snippet}", known=watchlist)
    hints = list(dict.fromkeys(hints + extracted))
    return Signal(
        url=url,
        title=title,
        source=source_label,
        published_at=published,
        snippet=snippet,
        tickers_hint=hints,
    )
