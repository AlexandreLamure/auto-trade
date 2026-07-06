"""NewsAPI.org source."""

from __future__ import annotations

import logging

import httpx

from news.models import Signal
from news.sources.base import parse_published
from news.sources.http_helpers import api_item_to_signal, valid_api_key

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"

MACRO_QUERIES = (
    "Federal Reserve interest rates stock market",
    "US stock market outlook",
    "sector rotation equities",
    "earnings report stocks",
    "SEC filing 8-K 10-Q",
)


async def fetch_newsapi(
    api_key: str,
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[Signal]:
    if not valid_api_key(api_key):
        return []

    watchlist = watchlist_tickers or []
    queries = list(MACRO_QUERIES)
    for sym in watchlist[:5]:
        queries.append(f"{sym} stock")

    articles: list[Signal] = []
    async with httpx.AsyncClient(timeout=15) as client:
        for query in queries:
            params = {
                "q": query,
                "pageSize": 10,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": api_key,
            }
            try:
                resp = await client.get(f"{NEWSAPI_BASE}/everything", params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("NewsAPI query failed (%s): %s", query, exc)
                continue

            for item in data.get("articles", []):
                source_name = (item.get("source") or {}).get("name") or "newsapi"
                signal = api_item_to_signal(
                    url=item.get("url") or "",
                    title=item.get("title") or "",
                    snippet=item.get("description") or "",
                    published=parse_published(item.get("publishedAt")),
                    source_label=f"api:newsapi:{source_name}",
                    watchlist=watchlist,
                )
                if signal:
                    articles.append(signal)
    return articles
