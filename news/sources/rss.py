"""RSS feed fetcher using feedparser."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import feedparser

from news.feeds import FeedConfig, build_feed_list
from news.models import RawArticle
from news.sources.base import extract_tickers, parse_published

logger = logging.getLogger(__name__)


def _parse_feed_sync(feed: FeedConfig, watchlist: list[str]) -> list[RawArticle]:
    articles: list[RawArticle] = []
    try:
        parsed = feedparser.parse(feed.url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RSS parse failed for %s: %s", feed.label, exc)
        return articles

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

        articles.append(
            RawArticle(
                url=url,
                title=title,
                source=f"rss:{feed.label}",
                published_at=published,
                snippet=snippet,
                tickers_hint=tickers,
            )
        )
    return articles


async def fetch_rss(
    watchlist_tickers: list[str],
    feeds: list[FeedConfig] | None = None,
) -> list[RawArticle]:
    feed_list = feeds or build_feed_list(watchlist_tickers)
    tasks = [
        asyncio.to_thread(_parse_feed_sync, feed, watchlist_tickers)
        for feed in feed_list
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[RawArticle] = []
    for feed, result in zip(feed_list, results):
        if isinstance(result, Exception):
            logger.warning("RSS fetch failed for %s: %s", feed.label, result)
            continue
        articles.extend(result)
    return articles
