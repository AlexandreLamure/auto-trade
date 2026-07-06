"""Alpha Vantage news sentiment source."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from news.models import Signal
from news.sources.base import extract_tickers, parse_published
from news.sources.http_helpers import api_item_to_signal, valid_api_key

logger = logging.getLogger(__name__)


def _parse_av_time(value: str | None) -> datetime | None:
    if not value or len(value) < 15:
        return None
    try:
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return parse_published(value)


async def fetch_alpha_vantage(
    api_key: str,
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[Signal]:
    if not valid_api_key(api_key):
        return []

    watchlist = watchlist_tickers or []
    tickers = ["FOREX:USD", "CRYPTO:BTC"] + [f"COIN:{s}" for s in watchlist[:3]]
    articles: list[Signal] = []

    async with httpx.AsyncClient(timeout=20) as client:
        for ticker in tickers:
            try:
                resp = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "NEWS_SENTIMENT",
                        "tickers": ticker,
                        "limit": 20,
                        "apikey": api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("Alpha Vantage news failed for %s: %s", ticker, exc)
                continue

            for item in data.get("feed", []):
                item_tickers = [
                    t.get("ticker", "").replace("COIN:", "").upper()
                    for t in item.get("ticker_sentiment", [])
                    if isinstance(t, dict)
                ]
                title = item.get("title") or ""
                snippet = item.get("summary") or title
                hints = list(
                    dict.fromkeys(
                        item_tickers + extract_tickers(f"{title} {snippet}", known=watchlist)
                    )
                )
                signal = api_item_to_signal(
                    url=item.get("url") or "",
                    title=title,
                    snippet=snippet,
                    published=_parse_av_time(item.get("time_published")),
                    source_label="api:alpha_vantage",
                    watchlist=watchlist,
                    tickers_hint=hints,
                )
                if signal:
                    articles.append(signal)
    return articles
