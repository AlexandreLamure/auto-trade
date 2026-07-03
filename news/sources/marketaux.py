"""Marketaux news source."""

from __future__ import annotations

import logging

import httpx

from news.models import RawArticle
from news.sources.base import extract_tickers, parse_published

logger = logging.getLogger(__name__)


async def fetch_marketaux(
    api_key: str,
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[RawArticle]:
    if not api_key or api_key.startswith("your_"):
        return []

    watchlist = watchlist_tickers or []
    params: dict[str, str | int] = {
        "api_token": api_key,
        "language": "en",
        "limit": 30,
    }
    if watchlist:
        params["symbols"] = ",".join(watchlist[:10])

    articles: list[RawArticle] = []
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get("https://api.marketaux.com/v1/news/all", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Marketaux news failed: %s", exc)
            return articles

        for item in data.get("data", []):
            url = item.get("url") or ""
            if not url:
                continue
            title = (item.get("title") or "").strip()
            snippet = (item.get("description") or title)[:500]
            symbols = [
                e.get("symbol", "").upper()
                for e in item.get("entities", [])
                if isinstance(e, dict) and e.get("symbol")
            ]
            hints = list(dict.fromkeys(symbols + extract_tickers(f"{title} {snippet}", known=watchlist)))

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    source="api:marketaux",
                    published_at=parse_published(item.get("published_at")),
                    snippet=snippet,
                    tickers_hint=hints,
                )
            )
    return articles
