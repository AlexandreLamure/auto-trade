"""
Deep research pipeline – programmatic MCP data collection for the committee.

Runs before deliberation: no LLM tool loop.  Produces a ResearchBrief that
every trader persona receives as shared context.

Portfolio and market data come from Alpaca MCP.  Market intelligence comes
from the shared event store (populated by the news analysis service).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import Settings
from servers.manager import MCPManager
from servers.portfolio import (
    Position,
    call_mcp_tool,
    estimate_days_held,
    format_holdings_table,
    load_positions_and_movers,
    parse_float_field,
    parse_mcp_json,
    parse_positions,
    unwrap_alpaca_payload,
)
from agent.personas import HORIZON_DAYS
from agent.workflow import (
    PriceStats,
    compute_price_analytics,
    compute_price_stats,
    expand_symbol_universe,
    filter_liquid_candidates,
    parse_latest_prices,
)
from store import (
    MarketEvent,
    StoredArticle,
    discover_event_tickers,
    get_event_articles,
    init_db,
    query_events,
)
from util.text import truncate_text

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
    price_analytics_markdown: str = ""
    events_markdown: str = ""
    raw_sections: dict[str, str] = field(default_factory=dict)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    market_events: list[MarketEvent] = field(default_factory=list)
    holdings: list[Position] = field(default_factory=list)
    holdings_markdown: str = ""
    price_stats: dict[str, PriceStats] = field(default_factory=dict)
    earnings_flags: dict[str, str] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        return self.summary_markdown

    def debate_context(self) -> str:
        """Abbreviated but evidence-rich context for Round 2 debate."""
        pnl = (
            f"{self.month_pnl_pct:+.2f}%"
            if self.month_pnl_pct is not None
            else "unavailable"
        )
        lines = [
            "## Portfolio",
            f"- Equity: ${self.portfolio_equity:,.2f} | Cash: ${self.cash_available:,.2f}",
            f"- Month PnL: {pnl}",
            f"- Holdings: {', '.join(self.held_symbols) or 'none'}",
            f"- Candidates: {', '.join(self.candidate_symbols) or 'none'}",
            "",
        ]
        if self.holdings_markdown.strip():
            lines.extend(["## Holdings", "", self.holdings_markdown, ""])
        if self.price_analytics_markdown.strip():
            lines.extend(["## Price analytics", "", self.price_analytics_markdown, ""])
        if self.events_markdown.strip():
            lines.extend(
                ["## Market events", "", truncate_text(self.events_markdown, 4000), ""]
            )
        return "\n".join(lines)


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


def _format_event(event: MarketEvent, articles: list[StoredArticle]) -> str:
    tickers = ", ".join(event.tickers) if event.tickers else "—"
    companies = ", ".join(event.companies) if event.companies else "—"
    updated = event.last_seen_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### {event.summary[:120]}{'…' if len(event.summary) > 120 else ''}",
        f"- Type: {event.event_type} | Sentiment: {event.sentiment} | "
        f"Importance: {event.importance}/5 | Confidence: {event.confidence:.0%}",
        f"- Tickers: {tickers} | Companies: {companies}",
        f"- Articles: {event.article_count} | Last updated: {updated}",
        f"- Summary: {truncate_text(event.summary, 800)}",
    ]
    if articles:
        lines.append("- Sources:")
        for article in articles[:2]:
            lines.append(
                f"  - [{article.source}, w={article.weight:.2f}] "
                f"{truncate_text(article.title, 100)}"
            )
    return "\n".join(lines)


def load_market_events(
    symbols: list[str],
    settings: Settings,
) -> tuple[str, list[MarketEvent]]:
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

    seen_ids = {e.id for e in symbol_events}
    macro_only = [e for e in macro_events if e.id not in seen_ids]

    lines: list[str] = []
    if macro_only:
        lines.append("### Macro / broad market")
        for event in macro_only:
            articles = get_event_articles(db_path, event.id)
            lines.append(_format_event(event, articles))
            lines.append("")

    if symbol_events:
        lines.append("### Symbol-specific events")
        for event in symbol_events:
            articles = get_event_articles(db_path, event.id)
            lines.append(_format_event(event, articles))
            lines.append("")

    if not lines:
        return "_No recent market events matched the current portfolio._", []

    body = "\n".join(lines)
    all_events = symbol_events + macro_only
    return body, all_events


def earnings_flags_from_events(
    events: list[MarketEvent],
    symbols: list[str],
    *,
    within_days: int,
) -> dict[str, str]:
    """Mark names with a recent earnings/guidance event in the lookback window."""
    from datetime import timedelta

    from util.time import utcnow

    allow = {s.upper() for s in symbols}
    cutoff = utcnow() - timedelta(days=within_days)
    flags: dict[str, str] = {}
    for event in events:
        kind = (event.event_type or "").lower().replace("-", "_")
        if kind not in ("earnings", "guidance"):
            continue
        seen = event.last_seen_at
        if seen.tzinfo is None:
            seen = seen.replace(tzinfo=timezone.utc)
        if seen < cutoff:
            continue
        for ticker in event.tickers:
            sym = ticker.upper()
            if sym in allow:
                flags[sym] = "recent"
    return flags


def drop_bearish_nonheld(
    candidates: list[str],
    held: list[str],
    events: list[MarketEvent],
    *,
    min_importance: int,
) -> tuple[list[str], list[str]]:
    """Remove non-held names that have a high-importance bearish event."""
    held_set = {s.upper() for s in held}
    blocked: list[str] = []
    blocked_set: set[str] = set()
    for event in events:
        if (event.sentiment or "").lower() != "bearish":
            continue
        if event.importance < min_importance:
            continue
        for ticker in event.tickers:
            sym = ticker.upper()
            if sym and sym not in held_set and sym not in blocked_set:
                blocked_set.add(sym)
                blocked.append(sym)
    kept = [c for c in candidates if c.upper() not in blocked_set]
    return kept, blocked


def _discover_symbols_from_events(settings: Settings, held: list[str]) -> list[str]:
    path = Path(settings.event_store_path)
    if not path.exists():
        return []
    init_db(settings.event_store_path)
    return discover_event_tickers(
        settings.event_store_path,
        since_hours=settings.events_since_hours,
        min_importance=settings.events_min_importance,
        limit=settings.event_discovery_limit,
        exclude=set(held),
    )


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
        "holdings": "Holdings",
        "price_analytics": "Price analytics",
        "account": "Account",
        "portfolio_history": f"Portfolio history ({HORIZON_DAYS}d)",
        "orders": "Recent orders",
        "movers": "Market movers",
        "market_events": "Market events",
    }
    for key, title in section_titles.items():
        body = sections.get(key, "").strip()
        if body:
            lines.extend([f"## {title}", "", truncate_text(body, 4000), ""])

    return "\n".join(lines)


async def run_deep_research(
    manager: MCPManager,
    settings: Settings,
) -> ResearchBrief:
    """Collect portfolio, market, and event-store data programmatically."""
    log: list[dict[str, Any]] = []
    sections: dict[str, str] = {}
    latest_prices: dict[str, float] = {}
    price_analytics_markdown = ""

    account = await call_mcp_tool(manager, "get_account_info", tool_call_log=log)
    sections["account"] = account
    cash, equity = _parse_account_cash(account)

    snapshot = await load_positions_and_movers(
        manager,
        n_movers=settings.mover_candidate_slots,
        min_price=settings.min_candidate_price,
        tool_call_log=log,
    )
    sections["positions"] = snapshot.positions_json
    held = snapshot.held

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
        {"status": "all", "limit": 100},
        tool_call_log=log,
    )
    sections["orders"] = orders

    holdings = parse_positions(snapshot.positions_json)
    days_held = estimate_days_held(orders)
    for pos in holdings:
        pos.days_held = days_held.get(pos.symbol)

    sections["movers"] = snapshot.movers_json
    mover_symbols = snapshot.mover_symbols

    event_tickers = _discover_symbols_from_events(settings, held)
    held, candidates = expand_symbol_universe(
        held,
        mover_symbols,
        event_tickers,
        max_candidates=settings.research_symbol_count,
        mover_slots=settings.mover_candidate_slots,
        event_slots=settings.event_candidate_slots,
    )
    all_symbols = list(dict.fromkeys(held + candidates))

    bars = ""
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
        latest_prices = parse_latest_prices(bars)
        if bars:
            candidates = filter_liquid_candidates(
                candidates,
                bars,
                min_price=settings.min_candidate_price,
                min_adv_shares=settings.min_adv_shares,
            )
            all_symbols = list(dict.fromkeys(held + candidates))

    events_text, market_events = load_market_events(all_symbols, settings)
    sections["market_events"] = events_text
    candidates, blocked_bearish = drop_bearish_nonheld(
        candidates,
        held,
        market_events,
        min_importance=settings.bearish_min_importance,
    )
    if blocked_bearish:
        sections["market_events"] = (
            events_text
            + "\n\n_Dropped non-held bearish names: "
            + ", ".join(blocked_bearish)
            + "_"
        )
    all_symbols = list(dict.fromkeys(held + candidates))
    earn_flags = earnings_flags_from_events(
        market_events,
        all_symbols,
        within_days=settings.earnings_blackout_days,
    )
    price_stats: dict[str, PriceStats] = {}
    if bars:
        price_stats = compute_price_stats(bars, all_symbols)
        price_analytics_markdown = compute_price_analytics(
            bars, all_symbols, earnings=earn_flags
        )
        sections["price_analytics"] = price_analytics_markdown

    for pos in holdings:
        if pos.current_price <= 0:
            pos.current_price = latest_prices.get(pos.symbol, 0.0)
        if pos.market_value <= 0 and pos.current_price and pos.qty:
            pos.market_value = pos.current_price * abs(pos.qty)
    holdings_markdown = format_holdings_table(holdings, equity=equity)
    sections["holdings"] = holdings_markdown

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
        price_analytics_markdown=price_analytics_markdown,
        events_markdown=events_text,
        raw_sections=sections,
        tool_call_log=log,
        market_events=market_events,
        holdings=holdings,
        holdings_markdown=holdings_markdown,
        price_stats=price_stats,
        earnings_flags=earn_flags,
    )


async def enrich_brief_prices(
    manager: MCPManager,
    brief: ResearchBrief,
    symbols: list[str],
    settings: Settings,
) -> ResearchBrief:
    """Fetch bar data for symbols missing from the research universe (chair verify)."""
    missing = list(
        dict.fromkeys(s.upper() for s in symbols if s.upper() not in brief.latest_prices)
    )
    if not missing:
        return brief

    bars = await call_mcp_tool(
        manager,
        "get_stock_bars",
        {
            "symbols": ",".join(missing),
            "timeframe": "1Day",
            "days": HORIZON_DAYS,
            "feed": "iex",
        },
        tool_call_log=brief.tool_call_log,
    )
    new_prices = parse_latest_prices(bars)
    extra_stats = compute_price_stats(bars)
    extra_analytics = compute_price_analytics(bars, earnings=brief.earnings_flags)

    updated_prices = {**brief.latest_prices, **new_prices}
    updated_stats = {**brief.price_stats, **extra_stats}

    updated_prices = {**brief.latest_prices, **new_prices}
    combined_analytics = brief.price_analytics_markdown
    if extra_analytics and "No price analytics" not in extra_analytics:
        combined_analytics = (
            f"{combined_analytics}\n\n### Additional symbols\n{extra_analytics}".strip()
        )

    all_symbols = list(dict.fromkeys(brief.all_symbols + missing))
    sections = dict(brief.raw_sections)
    sections["price_analytics"] = combined_analytics

    return replace(
        brief,
        all_symbols=all_symbols,
        latest_prices=updated_prices,
        price_analytics_markdown=combined_analytics,
        price_stats=updated_stats,
        raw_sections=sections,
    )
