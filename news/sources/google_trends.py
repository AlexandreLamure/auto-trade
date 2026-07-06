"""Google Trends spike detection via pytrends."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from news.models import Signal

logger = logging.getLogger(__name__)


def _fetch_trends_sync(
    watchlist: list[str],
    spike_threshold: float,
) -> list[Signal]:
    from pytrends.request import TrendReq

    if not watchlist:
        return []

    pytrends = TrendReq(hl="en-US", tz=360)
    signals: list[Signal] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Batch up to 5 keywords per request (pytrends limit)
    batch_size = 5
    for i in range(0, len(watchlist), batch_size):
        batch = watchlist[i : i + batch_size]
        try:
            pytrends.build_payload(batch, timeframe="today 3-m", geo="US")
            df = pytrends.interest_over_time()
        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "too many" in err:
                raise RuntimeError("Google Trends rate limited (429)") from exc
            logger.warning("Google Trends query failed for %s: %s", batch, exc)
            continue

        if df is None or df.empty:
            continue

        for keyword in batch:
            if keyword not in df.columns:
                continue
            series = df[keyword].dropna()
            if len(series) < 2:
                continue
            current = float(series.iloc[-1])
            trailing = series.iloc[:-1]
            avg = float(trailing.mean()) if len(trailing) else 0.0
            if avg <= 0 or current < avg * spike_threshold:
                continue

            sym = keyword.upper()
            url = (
                f"https://trends.google.com/trends/explore"
                f"?q={sym}&geo=US&date={today}"
            )
            title = f"{sym} Google Trends interest spiked to {current:.0f} (avg {avg:.0f})"
            snippet = (
                f"Search interest for {sym} is {current / avg:.1f}x above "
                f"its 3-month trailing average ({current:.0f} vs {avg:.0f})."
            )
            signals.append(
                Signal(
                    url=url,
                    title=title,
                    source="google_trends:spike",
                    published_at=datetime.now(timezone.utc),
                    snippet=snippet,
                    tickers_hint=[sym],
                )
            )

        if i + batch_size < len(watchlist):
            time.sleep(4)

    return signals


async def fetch_google_trends(
    *,
    watchlist_tickers: list[str] | None = None,
    spike_threshold: float = 1.5,
) -> list[Signal]:
    watchlist = watchlist_tickers or []
    if not watchlist:
        return []

    try:
        return await asyncio.to_thread(
            _fetch_trends_sync, watchlist, spike_threshold
        )
    except RuntimeError as exc:
        if "429" in str(exc):
            logger.warning("Google Trends rate limited, skipping this cycle")
        else:
            logger.warning("Google Trends failed: %s", exc)
        return []
    except Exception as exc:
        logger.warning("Google Trends failed: %s", exc)
        return []
