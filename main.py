"""
Entry point for the autonomous trading agent.

Starts an APScheduler async scheduler that triggers AgentOrchestrator.run_cycle()
on the configured interval (default: every 6 hours via LOOP_INTERVAL_HOURS).

Usage:
    python main.py              # runs on the configured schedule
    python main.py --once       # run a single cycle immediately and exit
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from config.settings import settings
from agent.orchestrator import AgentOrchestrator
from run_service import configure_stderr_logging, run_scheduled_service
from util.cycle_log import init_trading_log

LOGS_DIR = Path(__file__).resolve().parent / "logs"


async def _run_once(orchestrator: AgentOrchestrator) -> None:
    await orchestrator.run_cycle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Trading Agent")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis cycle and exit",
    )
    args = parser.parse_args()

    configure_stderr_logging(
        noisy_loggers=("httpx", "httpcore", "openai", "mcp", "anyio"),
    )
    log = init_trading_log(LOGS_DIR / "trading.log")

    log.line(
        f"Trading agent started | model={settings.ollama_model} | "
        f"interval={settings.loop_interval_hours}h | paper={settings.alpaca_paper_trade}"
    )

    orchestrator = AgentOrchestrator(settings)

    if args.once:
        asyncio.run(_run_once(orchestrator))
    else:
        asyncio.run(
            run_scheduled_service(
                job_fn=orchestrator.run_cycle,
                interval_hours=settings.loop_interval_hours,
                job_id="trading_cycle",
                job_name="Autonomous trading cycle",
            )
        )


if __name__ == "__main__":
    main()
