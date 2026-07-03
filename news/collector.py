"""Fetch articles from all configured news sources."""

from __future__ import annotations

import asyncio
import logging

from config.settings import Settings
from news.models import RawArticle
from news.sources.alpha_vantage import fetch_alpha_vantage
from news.sources.base import normalize_url, url_hash
from news.sources.finnhub import fetch_finnhub
from news.sources.marketaux import fetch_marketaux
from news.sources.newsapi import fetch_newsapi
from news.sources.rss import fetch_rss

logger = logging.getLogger(__name__)


def _parse_watchlist(settings: Settings) -> list[str]:
    raw = settings.watchlist_tickers.strip()
    if not raw:
        return []
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


async def collect_articles(settings: Settings) -> list[RawArticle]:
    """Fetch from all sources in parallel and dedupe by normalized URL."""
    watchlist = _parse_watchlist(settings)

    tasks = [
        fetch_rss(watchlist),
        fetch_newsapi(settings.newsapi_key, watchlist_tickers=watchlist),
        fetch_finnhub(settings.finnhub_api_key, watchlist_tickers=watchlist),
        fetch_alpha_vantage(settings.alpha_vantage_api_key, watchlist_tickers=watchlist),
        fetch_marketaux(settings.marketaux_api_key, watchlist_tickers=watchlist),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[RawArticle] = []
    seen_hashes: set[str] = set()

    for result in results:
        if isinstance(result, Exception):
            logger.warning("Source fetch failed: %s", result)
            continue
        for article in result:
            normalized = normalize_url(article.url)
            h = url_hash(normalized)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            articles.append(
                RawArticle(
                    url=normalized,
                    title=article.title,
                    source=article.source,
                    published_at=article.published_at,
                    snippet=article.snippet,
                    tickers_hint=article.tickers_hint,
                )
            )

    logger.info("Collected %d unique articles from all sources", len(articles))
    return articles
