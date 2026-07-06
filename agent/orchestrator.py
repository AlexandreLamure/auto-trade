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
from agent.llm_client import OllamaClient
from agent.decision import PortfolioDecision, TradeOrder
from agent.workflow import format_mcp_summary, is_market_open, truncate_text
from agent.deliberation import run_committee
from agent.research import ResearchBrief, run_deep_research
from agent.risk import build_validation_context, validate_orders
from store import MarketEvent
from util.cycle_log import CycleLog, get_trading_log

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
                market_open, clock_detail = await is_market_open(manager)
                if not market_open:
                    log.line(f"SKIP – {clock_detail}")
                    return
                await self._run_committee_cycle(manager, cycle_id, log)

        except Exception as exc:  # noqa: BLE001
            logger.error("Cycle failed: %s", exc, exc_info=True)
        finally:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            log.line(f"Duration: {elapsed:.0f}s")

    async def _run_committee_cycle(
        self, manager: MCPManager, cycle_id: str, log: CycleLog
    ) -> None:
        brief = await run_deep_research(manager, self._settings)
        _log_portfolio(brief, log)
        _log_mcp_calls(brief, log)
        _log_market_events(brief, self._settings, log)

        log.section("Deliberation")
        portfolio_decision, _transcript = await run_committee(
            self._llm,
            brief,
            self._settings,
            cycle_id=cycle_id,
            cycle_log=log,
        )
        _log_decision(portfolio_decision, log)

        ctx = build_validation_context(
            self._settings,
            cash_available=brief.cash_available,
            positions_json=brief.raw_sections.get("positions", ""),
            latest_prices=brief.latest_prices,
        )
        approved, rejections = validate_orders(portfolio_decision.orders, ctx)
        if rejections:
            logger.warning("Order validation notes: %s", "; ".join(rejections))

        log.section("Execution")
        if approved:
            await self._execute_portfolio(manager, approved, log)
        else:
            log.line("HOLD – no orders executed")

    async def _execute_portfolio(
        self,
        manager: MCPManager,
        orders: list[TradeOrder],
        log: CycleLog,
    ) -> None:
        for order in orders:
            try:
                result = await manager.call_tool(
                    "place_stock_order",
                    {
                        "symbol": order.symbol,
                        "side": order.side,
                        "qty": str(int(order.quantity)),
                        "type": "market",
                        "time_in_force": "day",
                    },
                )
                result_text = manager.result_to_text(result)
                summary = format_mcp_summary(
                    "place_stock_order",
                    {"symbol": order.symbol, "side": order.side, "qty": int(order.quantity)},
                    result_text,
                )
                log.line(
                    f"{order.side.upper()} {order.symbol} x{order.quantity:.0f} → submitted"
                )
                log.line(f"  {summary}")
            except Exception as exc:  # noqa: BLE001
                logger.error("Trade execution failed for %s: %s", order.symbol, exc)
                log.line(
                    f"{order.side.upper()} {order.symbol} x{order.quantity:.0f} → error"
                )


def _log_portfolio(brief: ResearchBrief, log: CycleLog) -> None:
    log.section("Portfolio")
    pnl = f"{brief.month_pnl_pct:+.1f}%" if brief.month_pnl_pct is not None else "unavailable"
    log.line(
        f"Equity ${brief.portfolio_equity:,.2f} | Cash ${brief.cash_available:,.2f} | "
        f"30d PnL {pnl}"
    )
    held_detail = []
    positions_raw = brief.raw_sections.get("positions", "")
    from agent.workflow import parse_mcp_json, unwrap_alpaca_payload, _iter_positions

    raw = parse_mcp_json(positions_raw)
    payload = unwrap_alpaca_payload(raw) if raw is not None else None
    positions = _iter_positions(payload)
    if positions:
        held_detail = [f"{s} {qty:g}" for s, qty in positions]
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
    elif not key.startswith("PK"):
        errors.append(
            f"ALPACA_API_KEY does not start with 'PK' (got '{key[:4]}...'). "
            "Paper trading keys start with PK. "
            "Get yours at https://app.alpaca.markets/paper/dashboard/overview"
        )

    if not secret or secret.startswith("your_"):
        errors.append("ALPACA_SECRET_KEY is not set (still has placeholder value).")

    if errors:
        msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"Configuration errors — fix your .env file:\n{msg}")
