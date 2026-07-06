"""Fetch signals from all configured sources."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from config.settings import Settings
from news.models import Signal
from news.sources.alpha_vantage import fetch_alpha_vantage
from news.sources.base import normalize_url, url_hash
from news.sources.finnhub import fetch_finnhub
from news.sources.google_trends import fetch_google_trends
from news.sources.ir_rss import fetch_ir_rss
from news.sources.marketaux import fetch_marketaux
from news.sources.newsapi import fetch_newsapi
from news.sources.polymarket import fetch_polymarket
from news.sources.reddit import fetch_reddit
from news.sources.rss import fetch_rss
from news.sources.sec_edgar import fetch_sec_edgar
from news.sources.stocktwits import fetch_stocktwits
from news.sources.http_helpers import valid_api_key
from news.weights import resolve_weight

logger = logging.getLogger(__name__)

SourceTask = tuple[str, Callable[[], Awaitable[list[Signal]]], bool]


def _build_source_tasks(settings: Settings, watchlist: list[str]) -> list[SourceTask]:
    """Return (name, fetch_coroutine_factory, enabled) for each source."""
    return [
        (
            "rss",
            lambda: fetch_rss(watchlist),
            True,
        ),
        (
            "newsapi",
            lambda: fetch_newsapi(settings.newsapi_key, watchlist_tickers=watchlist),
            valid_api_key(settings.newsapi_key),
        ),
        (
            "finnhub",
            lambda: fetch_finnhub(settings.finnhub_api_key, watchlist_tickers=watchlist),
            valid_api_key(settings.finnhub_api_key),
        ),
        (
            "alpha_vantage",
            lambda: fetch_alpha_vantage(
                settings.alpha_vantage_api_key, watchlist_tickers=watchlist
            ),
            valid_api_key(settings.alpha_vantage_api_key),
        ),
        (
            "marketaux",
            lambda: fetch_marketaux(settings.marketaux_api_key, watchlist_tickers=watchlist),
            valid_api_key(settings.marketaux_api_key),
        ),
        (
            "reddit",
            lambda: fetch_reddit(
                watchlist_tickers=watchlist,
                user_agent=settings.sec_user_agent,
            ),
            settings.enable_reddit,
        ),
        (
            "polymarket",
            lambda: fetch_polymarket(watchlist_tickers=watchlist),
            settings.enable_polymarket,
        ),
        (
            "google_trends",
            lambda: fetch_google_trends(
                watchlist_tickers=watchlist,
                spike_threshold=settings.google_trends_spike_threshold,
            ),
            settings.enable_google_trends,
        ),
        (
            "sec_edgar",
            lambda: fetch_sec_edgar(
                watchlist_tickers=watchlist,
                user_agent=settings.sec_user_agent,
            ),
            settings.enable_sec_edgar,
        ),
        (
            "ir_rss",
            lambda: fetch_ir_rss(
                ir_rss_feeds=settings.ir_rss_feeds,
                watchlist_tickers=watchlist,
            ),
            bool(settings.ir_rss_feeds.strip() and settings.ir_rss_feeds.strip() != "{}"),
        ),
        (
            "stocktwits",
            lambda: fetch_stocktwits(
                watchlist_tickers=watchlist,
                user_agent=settings.sec_user_agent,
            ),
            settings.enable_stocktwits,
        ),
    ]


async def collect_signals(
    settings: Settings, watchlist: list[str]
) -> tuple[list[Signal], dict[str, str]]:
    """Fetch from all enabled sources in parallel and dedupe by normalized URL."""
    weights = settings.source_weights_map

    tasks_config = _build_source_tasks(settings, watchlist)
    source_stats: dict[str, str] = {
        name: "disabled" for name, _, enabled in tasks_config if not enabled
    }
    active = [(name, fn) for name, fn, enabled in tasks_config if enabled]

    results = await asyncio.gather(
        *(fn() for _, fn in active),
        return_exceptions=True,
    )

    signals: list[Signal] = []
    seen_hashes: set[str] = set()

    for (name, _), result in zip(active, results):
        if isinstance(result, Exception):
            logger.warning("Source %s fetch failed: %s", name, result)
            source_stats[name] = "FAILED (see stderr)"
            continue
        source_stats[name] = f"{len(result)} signals"
        for signal in result:
            normalized = normalize_url(signal.url)
            h = url_hash(normalized)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            weight = resolve_weight(signal.source, weights)
            signals.append(
                Signal(
                    url=normalized,
                    title=signal.title,
                    source=signal.source,
                    published_at=signal.published_at,
                    snippet=signal.snippet,
                    tickers_hint=signal.tickers_hint,
                    weight=weight,
                )
            )

    return signals, source_stats
