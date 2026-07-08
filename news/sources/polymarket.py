"""Polymarket Gamma API source."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from news.models import Signal
from news.sources.base import extract_tickers, parse_published

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

FINANCE_KEYWORDS = frozenset(
    {
        "stock",
        "stocks",
        "earnings",
        "fed",
        "federal reserve",
        "interest rate",
        "s&p",
        "nasdaq",
        "dow",
        "market",
        "gdp",
        "inflation",
        "recession",
        "ipo",
        "merger",
        "sec",
        "tariff",
        "treasury",
        "nvda",
        "aapl",
        "tsla",
        "amzn",
        "msft",
        "googl",
    }
)


def _is_finance_related(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in FINANCE_KEYWORDS)


def _market_to_signal(market: dict, watchlist: list[str]) -> Signal | None:
    slug = market.get("slug") or market.get("conditionId") or ""
    question = (market.get("question") or market.get("title") or "").strip()
    if not slug or not question:
        return None

    description = (market.get("description") or "")[:300]
    outcomes = market.get("outcomePrices") or market.get("outcomes") or ""
    snippet_parts = [description] if description else []
    if outcomes:
        snippet_parts.append(f"Prices: {outcomes}")
    snippet = " | ".join(snippet_parts)[:500] or question[:500]

    end_date = market.get("endDate") or market.get("endDateIso")
    published = parse_published(end_date) if isinstance(end_date, str) else None
    now = datetime.now(timezone.utc)
    if published is None or published > now:
        published = now

    tickers = extract_tickers(f"{question} {snippet}", known=watchlist)
    if not tickers:
        return None
    if watchlist and not any(t in {s.upper() for s in watchlist} for t in tickers):
        return None
    url = f"https://polymarket.com/event/{slug}"

    return Signal(
        url=url,
        title=question,
        source="polymarket:market",
        published_at=published,
        snippet=snippet,
        tickers_hint=tickers,
    )


async def fetch_polymarket(
    *,
    watchlist_tickers: list[str] | None = None,
) -> list[Signal]:
    watchlist = watchlist_tickers or []
    signals: list[Signal] = []
    seen_slugs: set[str] = set()

    async with httpx.AsyncClient(timeout=20) as client:
        # Top finance-related markets by 24h volume
        try:
            resp = await client.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "closed": "false",
                    "limit": 50,
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            resp.raise_for_status()
            markets = resp.json()
            if isinstance(markets, list):
                for market in markets:
                    question = (market.get("question") or "").strip()
                    if not _is_finance_related(question):
                        continue
                    slug = market.get("slug") or ""
                    if slug in seen_slugs:
                        continue
                    sig = _market_to_signal(market, watchlist)
                    if sig:
                        seen_slugs.add(slug)
                        signals.append(sig)
        except httpx.HTTPError as exc:
            logger.warning("Polymarket markets fetch failed: %s", exc)

        # Search per watchlist ticker
        for sym in watchlist[:8]:
            try:
                resp = await client.get(
                    f"{GAMMA_BASE}/public-search",
                    params={"q": sym},
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("Polymarket search failed for %s: %s", sym, exc)
                continue

            events = data.get("events") or data.get("markets") or []
            if isinstance(data, list):
                events = data
            for item in events[:5]:
                market = item
                if "markets" in item and item["markets"]:
                    market = item["markets"][0]
                slug = market.get("slug") or ""
                if slug in seen_slugs:
                    continue
                sig = _market_to_signal(market, watchlist)
                if sig:
                    if sym.upper() not in sig.tickers_hint:
                        sig.tickers_hint = list(dict.fromkeys(sig.tickers_hint + [sym.upper()]))
                    seen_slugs.add(slug)
                    signals.append(sig)

    return signals
