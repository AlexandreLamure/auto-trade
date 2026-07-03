"""Alpha Vantage news sentiment source."""

from __future__ import annotations

import logging

from datetime import datetime, timezone

import httpx

from news.models import RawArticle
from news.sources.base import extract_tickers, parse_published


def _parse_av_time(value: str | None) -> datetime | None:
    if not value or len(value) < 15:
        return None
    try:
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return parse_published(value)

logger = logging.getLogger(__name__)


async def fetch_alpha_vantage(
    api_key: str,
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[RawArticle]:
    if not api_key or api_key.startswith("your_"):
        return []

    watchlist = watchlist_tickers or []
    tickers = ["FOREX:USD", "CRYPTO:BTC"] + [f"COIN:{s}" for s in watchlist[:3]]
    articles: list[RawArticle] = []

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
                url = item.get("url") or ""
                if not url:
                    continue
                title = (item.get("title") or "").strip()
                snippet = (item.get("summary") or title)[:500]
                item_tickers = [
                    t.get("ticker", "").replace("COIN:", "").upper()
                    for t in item.get("ticker_sentiment", [])
                    if isinstance(t, dict)
                ]
                hints = list(dict.fromkeys(item_tickers + extract_tickers(f"{title} {snippet}", known=watchlist)))

                articles.append(
                    RawArticle(
                        url=url,
                        title=title,
                        source="api:alpha_vantage",
                        published_at=_parse_av_time(item.get("time_published")),
                        snippet=snippet,
                        tickers_hint=hints,
                    )
                )
    return articles
