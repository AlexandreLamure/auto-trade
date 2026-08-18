"""Resolve the news watchlist from Alpaca portfolio, movers, and the event store."""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import Settings
from servers.manager import MCPManager
from servers.portfolio import load_positions_and_movers
from agent.workflow import expand_symbol_universe
from store import discover_event_tickers, init_db

logger = logging.getLogger(__name__)


async def resolve_watchlist(
    settings: Settings,
) -> tuple[list[str], list[str], list[str]]:
    """Return (watchlist, held, candidates) from Alpaca plus event-discovered names."""
    held: list[str] = []
    candidates: list[str] = []
    try:
        async with MCPManager(settings) as manager:
            snapshot = await load_positions_and_movers(
                manager,
                n_movers=settings.mover_candidate_slots,
                min_price=settings.min_candidate_price,
            )
            event_tickers: list[str] = []
            if Path(settings.event_store_path).exists():
                init_db(settings.event_store_path)
                event_tickers = discover_event_tickers(
                    settings.event_store_path,
                    since_hours=settings.events_since_hours,
                    min_importance=settings.events_min_importance,
                    limit=settings.event_discovery_limit,
                    exclude=set(snapshot.held),
                )
            held, candidates = expand_symbol_universe(
                snapshot.held,
                snapshot.mover_symbols,
                event_tickers,
                max_candidates=settings.research_symbol_count,
                mover_slots=settings.mover_candidate_slots,
                event_slots=settings.event_candidate_slots,
            )
            watchlist = list(dict.fromkeys(held + candidates))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not resolve watchlist from Alpaca: %s", exc)
        return [], [], []

    if not watchlist:
        logger.warning(
            "Empty watchlist – ticker-specific news skipped; macro RSS still runs"
        )
    return watchlist, held, candidates
