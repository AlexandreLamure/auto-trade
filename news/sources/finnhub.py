"""Finnhub market news source."""

from __future__ import annotations

import logging

from datetime import datetime, timezone

import httpx

from news.models import RawArticle
from news.sources.base import extract_tickers, parse_published

logger = logging.getLogger(__name__)


def _parse_finnhub_time(value: object | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return parse_published(str(value))


async def fetch_finnhub(
    api_key: str,
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[RawArticle]:
    if not api_key or api_key.startswith("your_"):
        return []

    watchlist = watchlist_tickers or []
    articles: list[RawArticle] = []

    async with httpx.AsyncClient(timeout=15) as client:
        # General market news
        try:
            resp = await client.get(
                "https://finnhub.io/api/v1/news",
                params={"category": "general", "token": api_key},
            )
            resp.raise_for_status()
            items = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Finnhub general news failed: %s", exc)
            items = []

        for item in items[:20]:
            url = item.get("url") or ""
            if not url:
                continue
            title = (item.get("headline") or "").strip()
            snippet = (item.get("summary") or title)[:500]
            related = item.get("related") or ""
            tickers = extract_tickers(f"{title} {snippet} {related}", known=watchlist)
            if related and isinstance(related, str):
                for sym in related.split(","):
                    s = sym.strip().upper()
                    if s and s not in tickers:
                        tickers.append(s)

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    source="api:finnhub",
                    published_at=_parse_finnhub_time(item.get("datetime")),
                    snippet=snippet,
                    tickers_hint=tickers,
                )
            )

        # Per-ticker company news
        for sym in watchlist[:5]:
            try:
                resp = await client.get(
                    "https://finnhub.io/api/v1/company-news",
                    params={
                        "symbol": sym,
                        "from": "2020-01-01",
                        "to": "2099-12-31",
                        "token": api_key,
                    },
                )
                resp.raise_for_status()
                items = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("Finnhub company news failed for %s: %s", sym, exc)
                continue

            for item in items[:5]:
                url = item.get("url") or ""
                if not url:
                    continue
                title = (item.get("headline") or "").strip()
                snippet = (item.get("summary") or title)[:500]
                articles.append(
                    RawArticle(
                        url=url,
                        title=title,
                        source=f"api:finnhub:{sym}",
                        published_at=_parse_finnhub_time(item.get("datetime")),
                        snippet=snippet,
                        tickers_hint=[sym.upper()],
                    )
                )
    return articles
