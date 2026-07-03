"""
Entry point for the news analysis service.

Runs every NEWS_LOOP_INTERVAL_HOURS (default 6), fetching news from
configured sources, grouping into market events, and storing in SQLite.

Usage:
    python news_main.py                  # continuous 6-hour schedule
    python news_main.py --once           # single cycle and exit
    python news_main.py --once --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings
from news.pipeline import run_cycle


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "openai", "feedparser"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _validate_news_settings() -> None:
    """Ensure at least RSS is available (always) and warn on missing API keys."""
    logger = logging.getLogger(__name__)
    has_api = any(
        key and not key.startswith("your_")
        for key in (
            settings.newsapi_key,
            settings.finnhub_api_key,
            settings.alpha_vantage_api_key,
            settings.marketaux_api_key,
        )
    )
    if not has_api:
        logger.warning(
            "No news API keys configured – running RSS feeds only. "
            "Set NEWSAPI_KEY, FINNHUB_API_KEY, etc. in .env for broader coverage."
        )
    logger.info("RSS feeds always enabled (zero cost)")


async def _run_once() -> None:
    await run_cycle(settings)


async def _run_scheduled() -> None:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_cycle,
        trigger=IntervalTrigger(hours=settings.news_loop_interval_hours),
        kwargs={"settings": settings},
        id="news_cycle",
        name="News analysis cycle",
        max_instances=1,
        misfire_grace_time=60,
    )

    stop_event = asyncio.Event()

    def _shutdown(*_: object) -> None:
        logging.getLogger(__name__).info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    scheduler.start()
    logging.getLogger(__name__).info(
        "News scheduler started – running every %d hour(s). Press Ctrl+C to stop.",
        settings.news_loop_interval_hours,
    )

    await run_cycle(settings)

    await stop_event.wait()
    scheduler.shutdown(wait=False)
    logging.getLogger(__name__).info("News scheduler stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="News Analysis Service")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    _validate_news_settings()

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting news service | model=%s | interval=%dh | store=%s",
        settings.ollama_model,
        settings.news_loop_interval_hours,
        settings.event_store_path,
    )

    if args.once:
        asyncio.run(_run_once())
    else:
        asyncio.run(_run_scheduled())


if __name__ == "__main__":
    main()
