"""
Entry point for the news analysis service.

Runs on the US cash session by default (09:20 and 12:50 ET weekdays),
fetching news from configured sources, grouping into market events, and
storing in SQLite. Set SESSION_SCHEDULE=false to use NEWS_LOOP_INTERVAL_HOURS.

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
from util.session import during_regular_hours, parse_session_times

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
        f"store={settings.event_store_path} | "
        f"schedule={'session ' + settings.news_session_times if settings.session_schedule else str(settings.news_loop_interval_hours) + 'h'}"
    )
    news_log.line(
        f"Sources: RSS, Reddit={settings.enable_reddit}, Polymarket={settings.enable_polymarket}, "
        f"GoogleTrends={settings.enable_google_trends}, SEC={settings.enable_sec_edgar}, "
        f"Stocktwits={settings.enable_stocktwits}"
    )

    if args.once:
        asyncio.run(_run_once())
    elif settings.session_schedule:
        cron_times = parse_session_times(settings.news_session_times)
        asyncio.run(
            run_scheduled_service(
                job_fn=run_cycle,
                cron_times=cron_times,
                timezone=settings.market_timezone,
                job_id="news_cycle",
                job_name="News analysis cycle",
                job_kwargs={"settings": settings},
                run_immediately=during_regular_hours(
                    settings.market_timezone, start_hour=8, end_hour=16
                ),
                misfire_grace_time=1800,
            )
        )
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
