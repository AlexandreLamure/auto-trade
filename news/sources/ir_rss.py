"""Company investor relations RSS feeds from IR_RSS_FEEDS config."""

from __future__ import annotations

import json
import logging

from news.feeds import FeedConfig
from news.models import Signal
from news.sources.rss import fetch_rss

logger = logging.getLogger(__name__)


def _parse_ir_feeds(raw: str) -> list[FeedConfig]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid IR_RSS_FEEDS JSON")
        return []
    if not isinstance(data, dict):
        return []

    feeds: list[FeedConfig] = []
    for ticker, url in data.items():
        sym = str(ticker).strip().upper()
        feed_url = str(url).strip()
        if sym and feed_url:
            feeds.append(FeedConfig(url=feed_url, label=f"ir_{sym.lower()}"))
    return feeds


async def fetch_ir_rss(
    *,
    ir_rss_feeds: str = "{}",
    watchlist_tickers: list[str] | None = None,
) -> list[Signal]:
    watchlist = watchlist_tickers or []
    feed_list = _parse_ir_feeds(ir_rss_feeds)
    if not feed_list:
        return []
    return await fetch_rss(watchlist, feeds=feed_list, source_prefix="ir")
