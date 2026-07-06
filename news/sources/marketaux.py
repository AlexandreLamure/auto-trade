"""Marketaux news source."""

from __future__ import annotations

import logging

import httpx

from news.models import Signal
from news.sources.base import extract_tickers, parse_published
from news.sources.http_helpers import api_item_to_signal, valid_api_key

logger = logging.getLogger(__name__)


async def fetch_marketaux(
    api_key: str,
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[Signal]:
    if not valid_api_key(api_key):
        return []

    watchlist = watchlist_tickers or []
    params: dict[str, str | int] = {
        "api_token": api_key,
        "language": "en",
        "limit": 30,
    }
    if watchlist:
        params["symbols"] = ",".join(watchlist[:10])

    articles: list[Signal] = []
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get("https://api.marketaux.com/v1/news/all", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Marketaux news failed: %s", exc)
            return articles

        for item in data.get("data", []):
            symbols = [
                e.get("symbol", "").upper()
                for e in item.get("entities", [])
                if isinstance(e, dict) and e.get("symbol")
            ]
            title = item.get("title") or ""
            snippet = item.get("description") or title
            hints = list(
                dict.fromkeys(symbols + extract_tickers(f"{title} {snippet}", known=watchlist))
            )
            signal = api_item_to_signal(
                url=item.get("url") or "",
                title=title,
                snippet=snippet,
                published=parse_published(item.get("published_at")),
                source_label="api:marketaux",
                watchlist=watchlist,
                tickers_hint=hints,
            )
            if signal:
                articles.append(signal)
    return articles
