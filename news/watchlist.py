"""Resolve the news watchlist from Alpaca portfolio and market movers."""

from __future__ import annotations

import logging

from config.settings import Settings
from agent.workflow import (
    call_mcp_tool,
    extract_mover_symbols,
    extract_position_symbols,
    merge_symbol_universe,
)
from servers.manager import MCPManager

logger = logging.getLogger(__name__)


async def resolve_watchlist(
    settings: Settings,
) -> tuple[list[str], list[str], list[str]]:
    """Return (watchlist, held, candidates) from Alpaca."""
    held: list[str] = []
    candidates: list[str] = []
    try:
        async with MCPManager(settings) as manager:
            positions = await call_mcp_tool(manager, "get_all_positions")
            held = extract_position_symbols(positions)

            movers = await call_mcp_tool(
                manager,
                "get_market_movers",
                {"market_type": "stocks"},
            )
            mover_symbols = extract_mover_symbols(
                movers, n=settings.research_symbol_count
            )

            held, candidates = merge_symbol_universe(
                held,
                mover_symbols,
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
