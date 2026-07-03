"""
Deep research pipeline – programmatic MCP data collection for the committee.

Runs before deliberation: no LLM tool loop.  Produces a ResearchBrief that
every trader persona receives as shared context.

Portfolio and market data come from Alpaca MCP.  Market intelligence comes
from the shared event store (populated by the news analysis service).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import Settings
from servers.manager import MCPManager
from agent.personas import HORIZON_DAYS
from agent.workflow import (
    call_mcp_tool,
    extract_mover_symbols,
    extract_position_symbols,
    merge_symbol_universe,
    parse_float_field,
    parse_latest_prices,
    parse_mcp_json,
    unwrap_alpaca_payload,
)
from store import MarketEvent, init_db, query_events

logger = logging.getLogger(__name__)


@dataclass
class ResearchBrief:
    timestamp: str
    held_symbols: list[str]
    candidate_symbols: list[str]
    all_symbols: list[str]
    cash_available: float
    portfolio_equity: float
    month_pnl_pct: float | None
    latest_prices: dict[str, float]
    summary_markdown: str
    raw_sections: dict[str, str] = field(default_factory=dict)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        return self.summary_markdown


def _parse_account_cash(account_json: str) -> tuple[float, float]:
    raw = parse_mcp_json(account_json)
    if not isinstance(raw, dict):
        return 0.0, 0.0
    payload = unwrap_alpaca_payload(raw)
    if not isinstance(payload, dict):
        return 0.0, 0.0
    cash = parse_float_field(payload.get("cash") or payload.get("buying_power"))
    equity = parse_float_field(
        payload.get("equity") or payload.get("portfolio_value") or cash
    )
    return cash, equity


def _parse_month_pnl(history_json: str) -> float | None:
    raw = parse_mcp_json(history_json)
    if not isinstance(raw, dict):
        return None
    payload = unwrap_alpaca_payload(raw)
    if not isinstance(payload, dict):
        return None

    equity = payload.get("equity") or payload.get("profit_loss") or []
    if not isinstance(equity, list) or len(equity) < 2:
        return None
    try:
        start = next((float(v) for v in equity if float(v) > 0), float(equity[0]))
        end = float(equity[-1])
    except (TypeError, ValueError):
        return None
    if start == 0:
        return None
    return ((end - start) / abs(start)) * 100.0


def _truncate(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [truncated]"


def _format_event(event: MarketEvent) -> str:
    tickers = ", ".join(event.tickers) if event.tickers else "—"
    companies = ", ".join(event.companies) if event.companies else "—"
    updated = event.last_seen_at.strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"### {event.summary[:120]}{'…' if len(event.summary) > 120 else ''}\n"
        f"- Type: {event.event_type} | Sentiment: {event.sentiment} | "
        f"Importance: {event.importance}/5 | Confidence: {event.confidence:.0%}\n"
        f"- Tickers: {tickers} | Companies: {companies}\n"
        f"- Articles: {event.article_count} | Last updated: {updated}\n"
        f"- Summary: {_truncate(event.summary, 800)}"
    )


def load_market_events(symbols: list[str], settings: Settings) -> tuple[str, list[MarketEvent]]:
    """Query the event store for symbol-specific and macro events."""
    db_path = settings.event_store_path
    path = Path(db_path)

    if not path.exists():
        logger.warning(
            "Event store not found at %s – run `python news_main.py --once` first",
            db_path,
        )
        return (
            "_No market events available. Start the news service: `python news_main.py`_",
            [],
        )

    init_db(db_path)

    symbol_events = query_events(
        db_path,
        symbols=symbols,
        since_hours=settings.events_since_hours,
        min_importance=settings.events_min_importance,
        limit=settings.events_limit,
    )

    macro_events = query_events(
        db_path,
        symbols=[],
        since_hours=settings.events_since_hours,
        min_importance=settings.macro_events_min_importance,
        limit=settings.macro_events_limit,
    )

    # Deduplicate: macro query may overlap symbol events
    seen_ids = {e.id for e in symbol_events}
    macro_only = [e for e in macro_events if e.id not in seen_ids]

    lines: list[str] = []
    if macro_only:
        lines.append("### Macro / broad market")
        for event in macro_only:
            lines.append(_format_event(event))
            lines.append("")

    if symbol_events:
        lines.append("### Symbol-specific events")
        for event in symbol_events:
            lines.append(_format_event(event))
            lines.append("")

    if not lines:
        return "_No recent market events matched the current portfolio._", []

    body = "\n".join(lines)
    all_events = symbol_events + macro_only
    return body, all_events


def _build_summary(
    *,
    held: list[str],
    candidates: list[str],
    cash: float,
    equity: float,
    month_pnl_pct: float | None,
    sections: dict[str, str],
) -> str:
    lines = [
        f"# Research Brief ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
        "",
        "## Objective",
        f"Maximize portfolio PnL over the next **{HORIZON_DAYS} days** (risk-adjusted).",
        "",
        "## Portfolio snapshot",
        f"- Equity: ${equity:,.2f}",
        f"- Cash / buying power: ${cash:,.2f}",
        f"- Month PnL: {month_pnl_pct:.2f}%" if month_pnl_pct is not None else "- Month PnL: unavailable",
        f"- Holdings: {', '.join(held) if held else '(none)'}",
        f"- Candidates: {', '.join(candidates) if candidates else '(none)'}",
        "",
    ]

    section_titles = {
        "account": "Account",
        "positions": "Open positions",
        "portfolio_history": f"Portfolio history ({HORIZON_DAYS}d)",
        "orders": "Recent orders",
        "movers": "Market movers",
        "bars": f"{HORIZON_DAYS}-day price bars",
        "market_events": "Market events",
    }
    for key, title in section_titles.items():
        body = sections.get(key, "").strip()
        if body:
            lines.extend([f"## {title}", "", _truncate(body, 4000), ""])

    return "\n".join(lines)


async def run_deep_research(
    manager: MCPManager,
    settings: Settings,
) -> ResearchBrief:
    """Collect portfolio, market, and event-store data programmatically."""
    log: list[dict[str, Any]] = []
    sections: dict[str, str] = {}
    latest_prices: dict[str, float] = {}

    account = await call_mcp_tool(manager, "get_account_info", tool_call_log=log)
    sections["account"] = account
    cash, equity = _parse_account_cash(account)

    positions = await call_mcp_tool(manager, "get_all_positions", tool_call_log=log)
    sections["positions"] = positions
    held = extract_position_symbols(positions)

    history = await call_mcp_tool(
        manager,
        "get_portfolio_history",
        {"period": "1M", "timeframe": "1D"},
        tool_call_log=log,
    )
    sections["portfolio_history"] = history
    month_pnl_pct = _parse_month_pnl(history)

    orders = await call_mcp_tool(
        manager,
        "get_orders",
        {"status": "all", "limit": 20},
        tool_call_log=log,
    )
    sections["orders"] = orders

    movers = await call_mcp_tool(
        manager,
        "get_market_movers",
        {"market_type": "stocks"},
        tool_call_log=log,
    )
    sections["movers"] = movers
    mover_symbols = extract_mover_symbols(movers, n=settings.research_symbol_count)

    held, candidates = merge_symbol_universe(
        held,
        mover_symbols,
        max_candidates=settings.research_symbol_count,
    )
    all_symbols = list(dict.fromkeys(held + candidates))

    if all_symbols:
        bars = await call_mcp_tool(
            manager,
            "get_stock_bars",
            {
                "symbols": ",".join(all_symbols),
                "timeframe": "1Day",
                "days": HORIZON_DAYS,
                "feed": "iex",
            },
            tool_call_log=log,
        )
        sections["bars"] = bars
        latest_prices = parse_latest_prices(bars)

    events_text, _ = load_market_events(all_symbols, settings)
    sections["market_events"] = events_text

    summary = _build_summary(
        held=held,
        candidates=candidates,
        cash=cash,
        equity=equity,
        month_pnl_pct=month_pnl_pct,
        sections=sections,
    )

    return ResearchBrief(
        timestamp=datetime.now(timezone.utc).isoformat(),
        held_symbols=held,
        candidate_symbols=candidates,
        all_symbols=all_symbols,
        cash_available=cash,
        portfolio_equity=equity,
        month_pnl_pct=month_pnl_pct,
        latest_prices=latest_prices,
        summary_markdown=summary,
        raw_sections=sections,
        tool_call_log=log,
    )
