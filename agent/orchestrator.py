"""
AgentOrchestrator – committee trading cycle.

Each call to run_cycle():
  1. Deep programmatic research (MCP tools)
  2. Multi-persona deliberation → PortfolioDecision
  3. Risk validation and multi-order execution
  4. Logs to logs/trading.log
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from config.settings import Settings
from servers.manager import MCPManager
from agent.decision import PortfolioDecision, TradeOrder
from agent.workflow import format_mcp_summary, format_order_qty, parse_fractionable, parse_nbbo, should_run_trading_cycle
from agent.deliberation import run_committee
from agent.research import ResearchBrief, enrich_brief_prices, run_deep_research
from agent.risk import (
    build_validation_context,
    inject_time_stops,
    stop_distance,
    validate_orders,
)
from store import MarketEvent
from util.cycle_log import CycleLog, get_trading_log
from util.llm_client import OllamaClient
from util.text import truncate_text

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            enable_thinking=settings.enable_thinking,
        )
        _validate_settings(settings)

    async def _refresh_news_if_stale(self, log: CycleLog) -> None:
        """Run a news cycle when the event store is missing or older than the stale window."""
        from datetime import timedelta
        from pathlib import Path

        from news.pipeline import run_cycle
        from store import init_db, latest_event_activity
        from util.time import utcnow

        path = Path(self._settings.event_store_path)
        max_age = timedelta(minutes=self._settings.news_stale_minutes)
        activity = None
        if path.exists():
            init_db(str(path))
            activity = latest_event_activity(str(path))
        if activity is not None:
            if activity.tzinfo is None:
                activity = activity.replace(tzinfo=timezone.utc)
            if utcnow() - activity <= max_age:
                log.line(
                    f"Event store fresh (last activity {activity.strftime('%Y-%m-%d %H:%M UTC')})"
                )
                return
        log.section("News refresh")
        if activity is None:
            log.line("Event store empty or missing – running news cycle first")
        else:
            log.line(
                f"Event store stale (last activity {activity.strftime('%Y-%m-%d %H:%M UTC')}) "
                "– running news cycle first"
            )
        await run_cycle(self._settings)

    async def run_cycle(self) -> None:
        """Execute one full analysis-and-trade cycle."""
        started_at = datetime.now(timezone.utc)
        cycle_id = started_at.strftime("%Y%m%dT%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
        log = get_trading_log()
        log.start_cycle(
            f"CYCLE {cycle_id} | {started_at.strftime('%Y-%m-%dT%H:%M:%SZ')} | "
            f"model={self._settings.ollama_model}"
        )

        try:
            async with MCPManager(self._settings) as manager:
                tradable, clock_detail = await should_run_trading_cycle(manager)
                if not tradable:
                    log.line(f"SKIP – {clock_detail}")
                    return
                await self._refresh_news_if_stale(log)
                await self._run_committee_cycle(manager, log)

        except Exception as exc:  # noqa: BLE001
            logger.error("Cycle failed: %s", exc, exc_info=True)
        finally:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            log.line(f"Duration: {elapsed:.0f}s")

    async def _run_committee_cycle(
        self, manager: MCPManager, log: CycleLog
    ) -> None:
        brief = await run_deep_research(manager, self._settings)
        _log_portfolio(brief, log)
        _log_mcp_calls(brief, log)
        _log_market_events(brief, self._settings, log)

        log.section("Deliberation")
        portfolio_decision = await run_committee(
            self._llm,
            brief,
            self._settings,
            cycle_log=log,
        )
        _log_decision(portfolio_decision, log)

        order_symbols = [o.symbol for o in portfolio_decision.orders]
        if order_symbols:
            brief = await enrich_brief_prices(
                manager, brief, order_symbols, self._settings
            )

        ctx = build_validation_context(
            self._settings,
            cash_available=brief.cash_available,
            positions_json=brief.raw_sections.get("positions", ""),
            latest_prices=brief.latest_prices,
            portfolio_equity=brief.portfolio_equity,
            price_stats=brief.price_stats,
            earnings_flags=brief.earnings_flags,
        )
        gated_orders = inject_time_stops(
            portfolio_decision.orders,
            brief.holdings,
            time_stop_days=self._settings.time_stop_days,
        )
        approved, rejections = validate_orders(gated_orders, ctx)
        if rejections:
            logger.warning("Order validation notes: %s", "; ".join(rejections))
            for note in rejections:
                log.line(f"Risk: {note}")

        log.section("Execution")
        if approved:
            await self._execute_portfolio(manager, approved, log, brief)
        else:
            log.line("HOLD – no orders executed")

    async def _execute_portfolio(
        self,
        manager: MCPManager,
        orders: list[TradeOrder],
        log: CycleLog,
        brief: ResearchBrief,
    ) -> None:
        for order in orders:
            try:
                quote_text = ""
                try:
                    quote_result = await manager.call_tool(
                        "get_stock_latest_quote", {"symbol": order.symbol}
                    )
                    quote_text = manager.result_to_text(quote_result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Quote fetch failed for %s: %s", order.symbol, exc)
                bid, ask = parse_nbbo(quote_text, order.symbol)
                last = brief.latest_prices.get(order.symbol, 0.0)
                if order.side == "buy":
                    limit_px = ask if ask > 0 else last
                else:
                    limit_px = bid if bid > 0 else last
                use_limit = limit_px > 0
                if use_limit:
                    limit_px = round(limit_px, 2)

                fractionable = False
                try:
                    asset_result = await manager.call_tool(
                        "get_asset", {"symbol": order.symbol}
                    )
                    fractionable = parse_fractionable(
                        manager.result_to_text(asset_result)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Asset lookup failed for %s: %s", order.symbol, exc)

                qty_str = format_order_qty(order.quantity, fractionable=fractionable)
                payload: dict[str, str] = {
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": qty_str,
                    "type": "limit" if use_limit else "market",
                    "time_in_force": "day",
                }
                if use_limit:
                    payload["limit_price"] = str(limit_px)
                if (
                    order.side == "buy"
                    and self._settings.enable_stops
                ):
                    ref = ask if ask > 0 else last
                    if ref > 0:
                        dist = stop_distance(
                            ref,
                            brief.price_stats.get(order.symbol),
                            stop_atr_multiple=self._settings.stop_atr_multiple,
                            stop_pct=self._settings.stop_pct,
                        )
                        stop_px = round(ref - dist, 2)
                        if 0 < stop_px < ref:
                            payload["stop_loss_stop_price"] = str(stop_px)
                result = await manager.call_tool("place_stock_order", payload)
                result_text = manager.result_to_text(result)
                failed = bool(result.isError) or result_text.startswith("ERROR")
                summary = format_mcp_summary(
                    "place_stock_order",
                    payload,
                    result_text,
                )
                status = "failed" if failed else "submitted"
                log.line(
                    f"{order.side.upper()} {order.symbol} x{qty_str} → {status}"
                )
                log.line(f"  {summary}")
                if use_limit and last > 0:
                    slip = (limit_px / last - 1.0) * 100.0
                    log.line(
                        f"  Slippage vs last close: limit ${limit_px:.2f} vs ${last:.2f} ({slip:+.2f}%)"
                    )
                if failed:
                    logger.warning(
                        "Order not filled for %s %s: %s",
                        order.side,
                        order.symbol,
                        result_text,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Trade execution failed for %s: %s", order.symbol, exc)
                log.line(
                    f"{order.side.upper()} {order.symbol} x{order.quantity:g} → error"
                )


def _log_portfolio(brief: ResearchBrief, log: CycleLog) -> None:
    log.section("Portfolio")
    pnl = f"{brief.month_pnl_pct:+.1f}%" if brief.month_pnl_pct is not None else "unavailable"
    log.line(
        f"Equity ${brief.portfolio_equity:,.2f} | Cash ${brief.cash_available:,.2f} | "
        f"30d PnL {pnl}"
    )
    if brief.holdings:
        held_detail = [
            f"{p.symbol} {p.qty:g} ({p.unrealized_plpc * 100:+.1f}%)"
            if abs(p.unrealized_plpc) <= 2
            else f"{p.symbol} {p.qty:g} ({p.unrealized_plpc:+.1f}%)"
            for p in brief.holdings
        ]
        log.line(f"Holdings: {', '.join(held_detail)}")
    else:
        log.line(
            f"Holdings: {', '.join(brief.held_symbols) if brief.held_symbols else '(none)'}"
        )
    if brief.candidate_symbols:
        log.line(f"Candidates: {', '.join(brief.candidate_symbols)}")


def _log_mcp_calls(brief: ResearchBrief, log: CycleLog) -> None:
    log.section("Alpaca MCP")
    for entry in brief.tool_call_log:
        log.line(
            format_mcp_summary(entry["name"], entry.get("args") or {}, entry["result"])
        )


def _log_market_events(brief: ResearchBrief, settings: Settings, log: CycleLog) -> None:
    log.section("Event store")
    events = brief.market_events
    if not events:
        log.line("No recent events matched the portfolio")
        return
    log.line(
        f"{len(events)} events (≥{settings.events_min_importance} importance, "
        f"{settings.events_since_hours}h):"
    )
    for event in events[:15]:
        log.line(_event_one_liner(event))
    if len(events) > 15:
        log.line(f"… and {len(events) - 15} more")


def _event_one_liner(event: MarketEvent) -> str:
    tickers = ", ".join(event.tickers) if event.tickers else "macro"
    summary = truncate_text(event.summary, 100)
    return f"[{tickers}] {summary} (importance {event.importance}/5)"


def _log_decision(decision: PortfolioDecision, log: CycleLog) -> None:
    log.section("Decision")
    if decision.is_trade():
        for order in decision.orders:
            log.line(f"Decision: {order.side.upper()} {order.symbol} x{order.quantity:.0f}")
    else:
        log.line("Decision: HOLD")
    if decision.consensus_summary:
        log.line(f"Consensus: {decision.consensus_summary}")
    if decision.dissent:
        log.line(f"Dissent: {decision.dissent}")


def _validate_settings(settings: Settings) -> None:
    """Fail fast with a clear message if credentials look wrong."""
    errors: list[str] = []

    key = settings.alpaca_api_key or ""
    secret = settings.alpaca_secret_key or ""

    if not key or key.startswith("your_"):
        errors.append("ALPACA_API_KEY is not set (still has placeholder value).")
    elif settings.alpaca_paper_trade and not key.startswith("PK"):
        errors.append(
            f"ALPACA_API_KEY does not start with 'PK' (got '{key[:4]}...'). "
            "Paper trading keys start with PK. "
            "Get yours at https://app.alpaca.markets/paper/dashboard/overview"
        )
    elif not settings.alpaca_paper_trade and not key.startswith("AK"):
        errors.append(
            f"ALPACA_API_KEY does not start with 'AK' (got '{key[:4]}...'). "
            "Live trading keys start with AK. "
            "Set ALPACA_PAPER_TRADE=true to use paper keys (PK…), or use live keys from "
            "https://app.alpaca.markets/live/dashboard/overview"
        )

    if not secret or secret.startswith("your_"):
        errors.append("ALPACA_SECRET_KEY is not set (still has placeholder value).")

    if errors:
        msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"Configuration errors — fix your .env file:\n{msg}")
