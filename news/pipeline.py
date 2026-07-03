"""News analysis pipeline – one cycle of fetch, analyze, store."""

from __future__ import annotations

import logging

from agent.llm_client import OllamaClient
from config.settings import Settings
from news.analyzer import process_articles
from news.collector import collect_articles
from store import count_events, init_db

logger = logging.getLogger(__name__)


async def run_cycle(settings: Settings) -> dict[str, int]:
    """Run one news collection and analysis cycle."""
    init_db(settings.event_store_path)

    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        enable_thinking=False,
    )

    articles = await collect_articles(settings)
    stats = await process_articles(settings, articles, llm)
    stats["total_events"] = count_events(settings.event_store_path)

    logger.info(
        "News cycle complete: fetched=%d new=%d skipped=%d events_updated=%d total_events=%d",
        stats.get("fetched", 0),
        stats.get("new", 0),
        stats.get("skipped", 0),
        stats.get("updated_events", 0),
        stats.get("total_events", 0),
    )
    return stats
