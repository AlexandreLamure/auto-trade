"""Finnhub market news source."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from news.models import Signal
from news.sources.base import extract_tickers, parse_published
from news.sources.http_helpers import api_item_to_signal, valid_api_key

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
) -> list[Signal]:
    if not valid_api_key(api_key):
        return []

    watchlist = watchlist_tickers or []
    articles: list[Signal] = []

    async with httpx.AsyncClient(timeout=15) as client:
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
            related = item.get("related") or ""
            tickers = extract_tickers(
                f"{item.get('headline', '')} {item.get('summary', '')} {related}",
                known=watchlist,
            )
            if related and isinstance(related, str):
                for sym in related.split(","):
                    s = sym.strip().upper()
                    if s and s not in tickers:
                        tickers.append(s)

            signal = api_item_to_signal(
                url=item.get("url") or "",
                title=item.get("headline") or "",
                snippet=item.get("summary") or "",
                published=_parse_finnhub_time(item.get("datetime")),
                source_label="api:finnhub",
                watchlist=watchlist,
                tickers_hint=tickers,
            )
            if signal:
                articles.append(signal)

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
                signal = api_item_to_signal(
                    url=item.get("url") or "",
                    title=item.get("headline") or "",
                    snippet=item.get("summary") or "",
                    published=_parse_finnhub_time(item.get("datetime")),
                    source_label=f"api:finnhub:{sym}",
                    watchlist=watchlist,
                    tickers_hint=[sym.upper()],
                )
                if signal:
                    articles.append(signal)
    return articles
