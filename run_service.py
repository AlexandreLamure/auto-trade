"""Shared APScheduler bootstrap for entry-point services."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger


def configure_stderr_logging(*, noisy_loggers: tuple[str, ...]) -> None:
    """Route warnings and errors to stderr only."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s – %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for noisy in noisy_loggers:
        logging.getLogger(noisy).setLevel(logging.ERROR)


async def run_scheduled_service(
    *,
    job_fn: Callable[..., Awaitable[None]],
    interval_hours: int,
    job_id: str,
    job_name: str,
    job_kwargs: dict[str, Any] | None = None,
    run_immediately: bool = True,
) -> None:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        job_fn,
        trigger=IntervalTrigger(hours=interval_hours),
        kwargs=job_kwargs or {},
        id=job_id,
        name=job_name,
        max_instances=1,
        misfire_grace_time=60,
    )

    stop_event = asyncio.Event()

    def _shutdown(*_: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    scheduler.start()

    if run_immediately:
        if job_kwargs:
            await job_fn(**job_kwargs)
        else:
            await job_fn()

    await stop_event.wait()
    scheduler.shutdown(wait=False)
