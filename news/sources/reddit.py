"""Reddit public JSON source (r/stocks, r/investing, r/wallstreetbets)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from news.models import Signal
from news.sources.base import extract_tickers

logger = logging.getLogger(__name__)

SUBREDDITS = ("stocks", "investing", "wallstreetbets")


async def fetch_reddit(
    *,
    watchlist_tickers: list[str] | None = None,
    user_agent: str = "auto-trade/1.0 (contact@example.com)",
) -> list[Signal]:
    watchlist = watchlist_tickers or []
    signals: list[Signal] = []
    headers = {"User-Agent": user_agent}

    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        for sub in SUBREDDITS:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=25"
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("Reddit fetch failed for r/%s: %s", sub, exc)
                continue

            for child in data.get("data", {}).get("children", []):
                post = child.get("data") or {}
                permalink = post.get("permalink") or ""
                if not permalink:
                    continue
                full_url = f"https://www.reddit.com{permalink}"
                title = (post.get("title") or "").strip()
                if not title:
                    continue
                selftext = (post.get("selftext") or "")[:500]
                snippet = selftext or title[:500]
                created = post.get("created_utc")
                published = (
                    datetime.fromtimestamp(created, tz=timezone.utc)
                    if created
                    else None
                )
                tickers = extract_tickers(f"{title} {snippet}", known=watchlist)

                signals.append(
                    Signal(
                        url=full_url,
                        title=title,
                        source=f"reddit:{sub}",
                        published_at=published,
                        snippet=snippet,
                        tickers_hint=tickers,
                    )
                )
    return signals
