"""RSS feed fetcher using feedparser."""

from __future__ import annotations

import asyncio
import logging

import feedparser

from news.feeds import FeedConfig, build_feed_list
from news.models import Signal
from news.sources.base import extract_tickers, parse_published

logger = logging.getLogger(__name__)


def parse_feed_entries(
    feed: FeedConfig,
    watchlist: list[str],
    *,
    source_prefix: str = "rss",
    source_label: str | None = None,
) -> list[Signal]:
    """Parse a single RSS/Atom feed into signals (sync, for use in thread pool)."""
    signals: list[Signal] = []
    try:
        parsed = feedparser.parse(feed.url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RSS parse failed for %s: %s", feed.label, exc)
        return signals

    for entry in parsed.entries[:30]:
        url = entry.get("link") or entry.get("id") or ""
        if not url:
            continue
        title = (entry.get("title") or "").strip()
        summary = entry.get("summary") or entry.get("description") or ""
        published = (
            parse_published(entry.get("published"))
            or parse_published(entry.get("updated"))
            or None
        )
        snippet = summary[:500] if summary else title[:500]
        tickers = extract_tickers(f"{title} {snippet}", known=watchlist)

        if source_label:
            source = source_label
        elif source_prefix == "ir":
            source = f"ir:{feed.label.replace('ir_', '').upper()}"
        else:
            source = f"rss:{feed.label}"

        signals.append(
            Signal(
                url=url,
                title=title,
                source=source,
                published_at=published,
                snippet=snippet,
                tickers_hint=tickers,
            )
        )
    return signals


def parse_atom_text(
    xml_text: str,
    *,
    source_label: str,
    watchlist: list[str],
) -> list[Signal]:
    """Parse Atom/RSS XML text into signals."""
    signals: list[Signal] = []
    try:
        parsed = feedparser.parse(xml_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Atom parse failed for %s: %s", source_label, exc)
        return signals

    for entry in parsed.entries[:30]:
        url = entry.get("link") or entry.get("id") or ""
        title = (entry.get("title") or "").strip()
        if not url or not title:
            continue
        summary = entry.get("summary") or entry.get("description") or ""
        snippet = summary[:500] if summary else title[:500]
        published = (
            parse_published(entry.get("published"))
            or parse_published(entry.get("updated"))
        )
        tickers = extract_tickers(f"{title} {snippet}", known=watchlist)
        signals.append(
            Signal(
                url=url,
                title=title,
                source=source_label,
                published_at=published,
                snippet=snippet,
                tickers_hint=tickers,
            )
        )
    return signals


async def fetch_rss(
    watchlist_tickers: list[str],
    feeds: list[FeedConfig] | None = None,
    *,
    source_prefix: str = "rss",
) -> list[Signal]:
    feed_list = feeds or build_feed_list(watchlist_tickers)
    tasks = [
        asyncio.to_thread(parse_feed_entries, feed, watchlist_tickers, source_prefix=source_prefix)
        for feed in feed_list
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    signals: list[Signal] = []
    for feed, result in zip(feed_list, results):
        if isinstance(result, Exception):
            logger.warning("RSS fetch failed for %s: %s", feed.label, result)
            continue
        signals.extend(result)
    return signals
