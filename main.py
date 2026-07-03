"""
Entry point for the autonomous trading agent.

Starts an APScheduler async scheduler that triggers AgentOrchestrator.run_cycle()
on the configured interval (default: every 60 minutes via LOOP_INTERVAL_MINUTES).

Usage:
    python main.py                        # runs on the configured schedule
    python main.py --once                 # run a single cycle immediately and exit
    python main.py --once --verbose       # print LLM thinking and decisions to stdout
    python main.py --once --log-level DEBUG
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
from agent.orchestrator import AgentOrchestrator


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "mcp", "anyio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _run_once(orchestrator: AgentOrchestrator) -> None:
    """Run exactly one cycle, then exit."""
    await orchestrator.run_cycle()


async def _run_scheduled(orchestrator: AgentOrchestrator) -> None:
    """Run the orchestrator on the configured interval until interrupted."""
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        orchestrator.run_cycle,
        trigger=IntervalTrigger(minutes=settings.loop_interval_minutes),
        id="trading_cycle",
        name="Autonomous trading cycle",
        max_instances=1,          # prevent overlapping runs
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
            # Windows does not support add_signal_handler for all signals
            signal.signal(sig, lambda *_: stop_event.set())

    scheduler.start()
    logging.getLogger(__name__).info(
        "Scheduler started – running every %d minute(s). Press Ctrl+C to stop.",
        settings.loop_interval_minutes,
    )

    # Run the first cycle immediately on startup
    await orchestrator.run_cycle()

    await stop_event.wait()
    scheduler.shutdown(wait=False)
    logging.getLogger(__name__).info("Scheduler stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Trading Agent")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis cycle and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print LLM thinking, tool calls and decision JSON to stdout",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting agent | model=%s | interval=%dm | paper=%s | thinking=%s",
        settings.ollama_model,
        settings.loop_interval_minutes,
        settings.alpaca_paper_trade,
        settings.enable_thinking,
    )

    orchestrator = AgentOrchestrator(settings, verbose=args.verbose)

    if args.once:
        asyncio.run(_run_once(orchestrator))
    else:
        asyncio.run(_run_scheduled(orchestrator))


if __name__ == "__main__":
    main()
