"""Stocktwits source (optional, disabled by default due to rate limits)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from news.models import Signal
from news.sources.base import extract_tickers

logger = logging.getLogger(__name__)

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"


async def fetch_stocktwits(
    *,
    watchlist_tickers: list[str] | None = None,
    user_agent: str = "auto-trade/1.0 (contact@example.com)",
) -> list[Signal]:
    watchlist = watchlist_tickers or []
    signals: list[Signal] = []
    headers = {"User-Agent": user_agent}
    failures = 0

    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        # Trending symbols
        try:
            resp = await client.get(f"{STOCKTWITS_BASE}/trending/symbols.json")
            if resp.status_code == 429:
                logger.warning("Stocktwits rate limited on trending endpoint")
                return []
            resp.raise_for_status()
            data = resp.json()
            for item in (data.get("symbols") or [])[:15]:
                symbol = (item.get("symbol") or "").strip()
                title = (item.get("title") or symbol).strip()
                if not symbol:
                    continue
                url = f"https://stocktwits.com/symbol/{symbol}"
                signals.append(
                    Signal(
                        url=url,
                        title=f"{symbol} trending on Stocktwits",
                        source="stocktwits:trending",
                        published_at=datetime.now(timezone.utc),
                        snippet=f"{title} is trending on Stocktwits.",
                        tickers_hint=[symbol.upper()],
                    )
                )
        except httpx.HTTPError as exc:
            logger.warning("Stocktwits trending fetch failed: %s", exc)
            failures += 1

        # Per-ticker streams (may require partner access)
        for sym in watchlist[:5]:
            try:
                resp = await client.get(
                    f"{STOCKTWITS_BASE}/streams/symbol/{sym}.json"
                )
                if resp.status_code == 429:
                    logger.warning("Stocktwits rate limited for %s", sym)
                    failures += 1
                    continue
                if resp.status_code == 403:
                    logger.debug("Stocktwits symbol stream requires partner access")
                    failures += 1
                    continue
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("Stocktwits stream failed for %s: %s", sym, exc)
                failures += 1
                continue

            for msg in (data.get("messages") or [])[:10]:
                body = (msg.get("body") or "").strip()
                msg_id = msg.get("id")
                if not body or not msg_id:
                    continue
                created = msg.get("created_at")
                published = None
                if created:
                    try:
                        published = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                    except ValueError:
                        published = datetime.now(timezone.utc)

                url = f"https://stocktwits.com/message/{msg_id}"
                tickers = extract_tickers(body, known=watchlist)
                if sym.upper() not in tickers:
                    tickers = list(dict.fromkeys(tickers + [sym.upper()]))

                signals.append(
                    Signal(
                        url=url,
                        title=body[:120],
                        source=f"stocktwits:{sym}",
                        published_at=published,
                        snippet=body[:500],
                        tickers_hint=tickers,
                    )
                )

    if failures and not signals:
        logger.warning("Stocktwits: all requests failed, returning empty")
    return signals
