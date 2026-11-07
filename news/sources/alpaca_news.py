"""Alpaca market news (Benzinga and partners), using the trading account's keys."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from news.models import Signal
from news.sources.base import parse_published
from news.sources.http_helpers import api_item_to_signal, valid_api_key
from util.symbols import is_tradeable_equity_symbol

logger = logging.getLogger(__name__)

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
LOOKBACK_HOURS = 48
PAGE_LIMIT = 50  # API maximum
MAX_PAGES = 3
# Wrap-up articles are tagged with dozens of symbols; keep only watchlist ones.
MAX_UNFILTERED_SYMBOLS = 5


async def _fetch_pages(
    client: httpx.AsyncClient, params: dict[str, str | int]
) -> list[dict]:
    items: list[dict] = []
    token = ""
    for _ in range(MAX_PAGES):
        page_params = dict(params)
        if token:
            page_params["page_token"] = token
        resp = await client.get(NEWS_URL, params=page_params)
        resp.raise_for_status()
        data = resp.json()
        items.extend(n for n in data.get("news") or [] if isinstance(n, dict))
        token = data.get("next_page_token") or ""
        if not token:
            break
    return items


def _tickers(item: dict, watchset: set[str]) -> list[str]:
    symbols = [
        str(s).upper() for s in item.get("symbols") or [] if is_tradeable_equity_symbol(str(s))
    ]
    if len(symbols) > MAX_UNFILTERED_SYMBOLS:
        symbols = [s for s in symbols if s in watchset]
    return symbols


async def fetch_alpaca_news(
    api_key: str,
    secret_key: str,
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[Signal]:
    if not (valid_api_key(api_key) and valid_api_key(secret_key)):
        return []

    watchlist = watchlist_tickers or []
    watchset = {s.upper() for s in watchlist}
    start = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    base: dict[str, str | int] = {
        "start": start,
        "limit": PAGE_LIMIT,
        "sort": "desc",
        "exclude_contentless": "true",
    }
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

    items: list[dict] = []
    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        # Watchlist news first, then the general market feed.
        queries = [{**base, "symbols": ",".join(watchlist)}] if watchlist else []
        queries.append(base)
        for params in queries:
            try:
                items.extend(await _fetch_pages(client, params))
            except httpx.HTTPError as exc:
                logger.warning("Alpaca news fetch failed: %s", exc)

    signals: list[Signal] = []
    seen: set[str] = set()
    for item in items:
        item_id = str(item.get("id") or item.get("url") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        signal = api_item_to_signal(
            url=item.get("url") or "",
            title=item.get("headline") or "",
            snippet=item.get("summary") or "",
            published=parse_published(str(item.get("created_at") or "")),
            source_label=f"api:alpaca:{item.get('source') or 'news'}",
            watchlist=watchlist,
            tickers_hint=_tickers(item, watchset),
        )
        if signal:
            signals.append(signal)
    return signals
