"""Single entry point for news and trading cycles.

By default the process does nothing. Pick exactly one mode:

    python main.py --news-once     # one news cycle, then exit
    python main.py --trade-once    # one trade cycle, then exit
    python main.py --once          # news then trade (sequential), then exit
    python main.py --loop          # news then trade every day at 12:00 America/New_York
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from agent.orchestrator import AgentOrchestrator
from config.settings import settings
from news.pipeline import run_cycle as run_news_cycle
from news.sources.http_helpers import valid_api_key
from util.cycle_log import CycleLog, init_news_log, init_trading_log
from util.session import MARKET_TZ, now_et

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOOP_HOUR = 12
LOOP_MINUTE = 0


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s – %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai", "mcp", "anyio", "feedparser"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _warn_if_no_news_keys() -> None:
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
        logging.getLogger(__name__).warning(
            "No news API keys configured – running RSS and free signal sources only. "
            "Set NEWSAPI_KEY, FINNHUB_API_KEY, etc. in .env for broader coverage."
        )


def _seconds_until_next_loop(last_run_date: date | None = None) -> tuple[float, datetime]:
    now = now_et()
    target = now.replace(hour=LOOP_HOUR, minute=LOOP_MINUTE, second=0, microsecond=0)
    if now > target or last_run_date == target.date():
        target += timedelta(days=1)
    return max(0.0, (target - now).total_seconds()), target


def _log_wait(logs: list[CycleLog], target: datetime, delay: float) -> None:
    hours = delay / 3600.0
    msg = f"Next cycle at {target.strftime('%Y-%m-%d %H:%M %Z')} (in {hours:.1f}h)"
    for log in logs:
        log.line(msg)


async def _loop(news_log: CycleLog, trade_log: CycleLog, orchestrator: AgentOrchestrator) -> None:
    logs = [news_log, trade_log]
    last_run_date: date | None = None
    while True:
        delay, target = _seconds_until_next_loop(last_run_date)
        _log_wait(logs, target, delay)
        await asyncio.sleep(delay)
        await run_news_cycle(settings)
        await orchestrator.run_cycle()
        last_run_date = now_et().date()


def main() -> None:
    parser = argparse.ArgumentParser(description="News analysis and trading agent")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--news-once", action="store_true", help="Run one news cycle and exit")
    mode.add_argument("--trade-once", action="store_true", help="Run one trade cycle and exit")
    mode.add_argument(
        "--once",
        action="store_true",
        help="Run one news cycle then one trade cycle and exit",
    )
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Run news then trade every day at 12:00 America/New_York",
    )
    args = parser.parse_args()

    if not (args.news_once or args.trade_once or args.once or args.loop):
        parser.print_help()
        return

    _configure_logging()

    run_news = args.news_once or args.once or args.loop
    run_trade = args.trade_once or args.once or args.loop
    news_log = init_news_log(LOGS_DIR / "news.log") if run_news else None
    trade_log = init_trading_log(LOGS_DIR / "trading.log") if run_trade else None

    if run_news:
        _warn_if_no_news_keys()
        assert news_log is not None
        news_log.line(
            f"News started | model={settings.ollama_model} | store={settings.event_store_path}"
        )

    orchestrator: AgentOrchestrator | None = None
    if run_trade:
        orchestrator = AgentOrchestrator(settings)
        assert trade_log is not None
        trade_log.line(
            f"Trading started | model={settings.ollama_model} | "
            f"paper={settings.alpaca_paper_trade}"
        )

    if args.news_once:
        asyncio.run(run_news_cycle(settings))
        return
    if args.trade_once:
        assert orchestrator is not None
        asyncio.run(orchestrator.run_cycle())
        return
    if args.once:
        assert orchestrator is not None
        asyncio.run(_run_once(orchestrator))
        return

    assert news_log is not None and trade_log is not None and orchestrator is not None
    news_log.line(f"Loop: news then trade daily at {LOOP_HOUR:02d}:{LOOP_MINUTE:02d} {MARKET_TZ}")
    trade_log.line(f"Loop: news then trade daily at {LOOP_HOUR:02d}:{LOOP_MINUTE:02d} {MARKET_TZ}")
    asyncio.run(_loop(news_log, trade_log, orchestrator))


async def _run_once(orchestrator: AgentOrchestrator) -> None:
    await run_news_cycle(settings)
    await orchestrator.run_cycle()


if __name__ == "__main__":
    main()
