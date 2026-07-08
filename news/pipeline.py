"""News analysis pipeline – one cycle of fetch, analyze, store."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from agent.llm_client import OllamaClient
from config.settings import Settings
from news.analyzer import process_signals
from news.collector import collect_signals
from news.watchlist import resolve_watchlist
from store import count_events, init_db, prune_old_events
from util.cycle_log import get_news_log


async def run_cycle(settings: Settings) -> dict[str, int]:
    """Run one news collection and analysis cycle."""
    log = get_news_log()
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.start_cycle(f"NEWS CYCLE {started_at}")

    init_db(settings.event_store_path)
    pruned = prune_old_events(settings.event_store_path, ttl_days=settings.event_ttl_days)
    if pruned:
        log.line(f"Pruned {pruned} events older than {settings.event_ttl_days}d")

    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        enable_thinking=False,
    )

    watchlist, held, candidates = await resolve_watchlist(settings)
    log.section("Watchlist")
    log.line(
        f"Held: {', '.join(held) or 'none'} | "
        f"Candidates: {', '.join(candidates) or 'none'} | "
        f"Total: {len(watchlist)}"
    )

    signals, source_stats = await collect_signals(settings, watchlist)
    log.section("Sources")
    for name, stat in source_stats.items():
        log.line(f"{name}: {stat}")

    log.section("Collection")
    active_count = sum(1 for s in source_stats.values() if s != "disabled")
    log.line(f"{len(signals)} unique signals from {active_count} active sources")

    stats = await process_signals(settings, signals, llm, watchlist, log)
    stats["total_events"] = count_events(settings.event_store_path)

    elapsed = time.monotonic() - started
    log.section("Summary")
    log.line(
        f"new={stats.get('new', 0)} skipped={stats.get('skipped', 0)} "
        f"events_updated={stats.get('updated_events', 0)} "
        f"total_events={stats.get('total_events', 0)} | duration={elapsed:.0f}s"
    )
    return stats
