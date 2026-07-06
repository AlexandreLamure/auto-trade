"""
Entry point for the news analysis service.

Runs every NEWS_LOOP_INTERVAL_HOURS (default 6), fetching news from
configured sources, grouping into market events, and storing in SQLite.

Usage:
    python news_main.py              # continuous 6-hour schedule
    python news_main.py --once       # single cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from config.settings import settings
from news.pipeline import run_cycle
from run_service import configure_stderr_logging, run_scheduled_service
from util.cycle_log import init_news_log

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _validate_news_settings() -> None:
    """Ensure at least RSS is available (always) and warn on missing API keys."""
    logger = logging.getLogger(__name__)
    from news.sources.http_helpers import valid_api_key

    has_api = any(
        valid_api_key(key)
        for key in (
            settings.newsapi_key,
            settings.finnhub_api_key,
            settings.alpha_vantage_api_key,
            settings.marketaux_api_key,
        )
    )
    if not has_api:
        logger.warning(
            "No news API keys configured – running RSS and free signal sources only. "
            "Set NEWSAPI_KEY, FINNHUB_API_KEY, etc. in .env for broader coverage."
        )


async def _run_once() -> None:
    await run_cycle(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="News Analysis Service")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    configure_stderr_logging(
        noisy_loggers=("httpx", "httpcore", "openai", "feedparser"),
    )
    news_log = init_news_log(LOGS_DIR / "news.log")
    _validate_news_settings()

    news_log.line(
        f"News service started | model={settings.ollama_model} | "
        f"interval={settings.news_loop_interval_hours}h | store={settings.event_store_path}"
    )
    news_log.line(
        f"Sources: RSS, Reddit={settings.enable_reddit}, Polymarket={settings.enable_polymarket}, "
        f"GoogleTrends={settings.enable_google_trends}, SEC={settings.enable_sec_edgar}, "
        f"Stocktwits={settings.enable_stocktwits}"
    )

    if args.once:
        asyncio.run(_run_once())
    else:
        asyncio.run(
            run_scheduled_service(
                job_fn=run_cycle,
                interval_hours=settings.news_loop_interval_hours,
                job_id="news_cycle",
                job_name="News analysis cycle",
                job_kwargs={"settings": settings},
            )
        )


if __name__ == "__main__":
    main()
