"""Shared APScheduler bootstrap for entry-point services."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
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


def _cron_trigger(
    hour: int,
    minute: int,
    *,
    timezone: str,
    weekdays_only: bool,
) -> CronTrigger:
    return CronTrigger(
        day_of_week="mon-fri" if weekdays_only else "*",
        hour=hour,
        minute=minute,
        timezone=ZoneInfo(timezone),
    )


async def run_scheduled_service(
    *,
    job_fn: Callable[..., Awaitable[Any]],
    interval_hours: int | None = None,
    cron_times: list[tuple[int, int]] | None = None,
    timezone: str = "America/New_York",
    weekdays_only: bool = True,
    job_id: str,
    job_name: str,
    job_kwargs: dict[str, Any] | None = None,
    run_immediately: bool = True,
    misfire_grace_time: int = 60,
) -> None:
    scheduler = AsyncIOScheduler(timezone=ZoneInfo(timezone))
    kwargs = job_kwargs or {}

    if cron_times:
        for index, (hour, minute) in enumerate(cron_times):
            scheduler.add_job(
                job_fn,
                trigger=_cron_trigger(
                    hour, minute, timezone=timezone, weekdays_only=weekdays_only
                ),
                kwargs=kwargs,
                id=f"{job_id}_{index}",
                name=f"{job_name} @ {hour:02d}:{minute:02d}",
                max_instances=1,
                misfire_grace_time=misfire_grace_time,
            )
    else:
        hours = interval_hours if interval_hours is not None else 6
        scheduler.add_job(
            job_fn,
            trigger=IntervalTrigger(hours=hours),
            kwargs=kwargs,
            id=job_id,
            name=job_name,
            max_instances=1,
            misfire_grace_time=misfire_grace_time,
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
        if kwargs:
            await job_fn(**kwargs)
        else:
            await job_fn()

    await stop_event.wait()
    scheduler.shutdown(wait=False)
