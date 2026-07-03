"""NewsAPI.org source."""

from __future__ import annotations

import logging

import httpx

from news.models import RawArticle
from news.sources.base import extract_tickers, parse_published

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
) -> list[RawArticle]:
    if not api_key or api_key.startswith("your_"):
        return []

    watchlist = watchlist_tickers or []
    queries = list(MACRO_QUERIES)
    for sym in watchlist[:5]:
        queries.append(f"{sym} stock")

    articles: list[RawArticle] = []
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
                url = item.get("url") or ""
                if not url:
                    continue
                title = (item.get("title") or "").strip()
                description = (item.get("description") or "").strip()
                source_name = (item.get("source") or {}).get("name") or "newsapi"
                published = parse_published(item.get("publishedAt"))
                snippet = description or title
                tickers = extract_tickers(f"{title} {snippet}", known=watchlist)

                articles.append(
                    RawArticle(
                        url=url,
                        title=title,
                        source=f"api:newsapi:{source_name}",
                        published_at=published,
                        snippet=snippet[:500],
                        tickers_hint=tickers,
                    )
                )
    return articles
