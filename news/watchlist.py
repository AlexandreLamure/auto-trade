"""Resolve the news watchlist from Alpaca portfolio and market movers."""

from __future__ import annotations

import logging

from config.settings import Settings
from servers.manager import MCPManager
from servers.portfolio import load_positions_and_movers, merge_symbol_universe

logger = logging.getLogger(__name__)


async def resolve_watchlist(
    settings: Settings,
) -> tuple[list[str], list[str], list[str]]:
    """Return (watchlist, held, candidates) from Alpaca."""
    held: list[str] = []
    candidates: list[str] = []
    try:
        async with MCPManager(settings) as manager:
            snapshot = await load_positions_and_movers(
                manager, n_movers=settings.research_symbol_count
            )
            held, candidates = merge_symbol_universe(
                snapshot.held,
                snapshot.mover_symbols,
                max_candidates=settings.research_symbol_count,
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
