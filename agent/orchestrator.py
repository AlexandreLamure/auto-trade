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
from agent.decision import PortfolioDecision, TradeOrder, order_to_dict, proposal_to_dict
from agent.workflow import (
    ExecutionRecord,
    format_mcp_summary,
    format_order_qty,
    parse_fractionable,
    parse_nbbo,
    parse_order_status,
)
from agent.deliberation import run_committee
from agent.research import ResearchBrief, enrich_brief_prices, run_deep_research
from agent.risk import (
    build_validation_context,
    stop_distance,
    validate_orders,
)
from store import MarketEvent
from store.journal import record_cycle, update_marks
from util.cycle_log import CycleLog, get_trading_log
from util.llm_client import OllamaClient
from util.session import is_market_open, market_closed_message

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            enable_thinking=settings.enable_thinking,
            num_ctx=settings.ollama_num_ctx,
            temperature=settings.ollama_temperature,
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
            if not is_market_open():
                log.line(market_closed_message())
                return
            async with MCPManager(self._settings) as manager:
                await self._run_committee_cycle(manager, log, cycle_id)

        except Exception as exc:  # noqa: BLE001
            logger.error("Cycle failed: %s", exc, exc_info=True)
        finally:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            log.line(f"Duration: {elapsed:.0f}s")

    async def _run_committee_cycle(
        self, manager: MCPManager, log: CycleLog, cycle_id: str
    ) -> None:
        brief = await run_deep_research(manager, self._settings)
        try:
            marked = update_marks(self._settings.event_store_path, brief.latest_prices)
            if marked:
                log.line(f"Journal: updated {marked} outcome mark(s)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Journal mark update failed: %s", exc)
        _log_portfolio(brief, log)
        _log_mcp_calls(brief, log)
        _log_market_events(brief, self._settings, log)

        log.section("Deliberation")
        portfolio_decision, proposals = await run_committee(
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
        approved, rejections = validate_orders(gated_orders, ctx)
        if rejections:
            logger.warning("Order validation notes: %s", "; ".join(rejections))
            for note in rejections:
                log.line(f"Risk: {note}")

        fills: list[ExecutionRecord] = []
        log.section("Execution")
        if approved:
            fills = await self._execute_portfolio(manager, approved, log, brief)
        else:
            log.line("HOLD – no orders executed")

        try:
            record_cycle(
                self._settings.event_store_path,
                cycle_id=cycle_id,
                equity=brief.portfolio_equity,
                cash=brief.cash_available,
                month_pnl_pct=brief.month_pnl_pct,
                consensus=portfolio_decision.consensus_summary,
                dissent=portfolio_decision.dissent,
                markdown=brief.summary_markdown,
                proposals=[proposal_to_dict(p) for p in proposals],
                proposed=[order_to_dict(o) for o in gated_orders],
                approved_keys={(o.symbol, o.side) for o in approved},
                rejections=rejections,
                fill_status={(rec.symbol, rec.side): rec.status for rec in fills},
                fill_records=[
                    {
                        "symbol": rec.symbol,
                        "side": rec.side,
                        "qty": rec.submitted_qty,
                        "status": rec.status,
                        "limit_price": rec.limit_price,
                        "reason": rec.reason,
                    }
                    for rec in fills
                ],
                prices=brief.latest_prices,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist decision journal: %s", exc)

    async def _execute_portfolio(
        self,
        manager: MCPManager,
        orders: list[TradeOrder],
        log: CycleLog,
        brief: ResearchBrief,
    ) -> list[ExecutionRecord]:
        fills: list[ExecutionRecord] = []
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
                log.line(
                    f"  Quote {order.symbol}: bid={bid or '—'} ask={ask or '—'} last={last or '—'}"
                )

                use_limit = bid > 0 and ask > 0
                limit_px = 0.0
                if use_limit:
                    if order.side == "buy":
                        limit_px = round(ask * 1.001, 2)
                    else:
                        limit_px = round(bid * 0.999, 2)
                    if limit_px <= 0:
                        use_limit = False

                fractionable = False
                try:
                    asset_result = await manager.call_tool(
                        "get_asset", {"symbol": order.symbol}
                    )
                    asset_text = manager.result_to_text(asset_result)
                    fractionable = parse_fractionable(asset_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Asset lookup failed for %s: %s", order.symbol, exc)
                log.line(f"  Asset {order.symbol}: fractionable={fractionable}")

                qty_str = format_order_qty(order.quantity, fractionable=fractionable)
                if qty_str is None:
                    reason = (
                        f"{order.side.upper()} {order.symbol}: skipped, "
                        f"qty {order.quantity:g} < 1 and not fractionable "
                        "(will not round up past the risk cap)"
                    )
                    log.line(reason)
                    fills.append(
                        ExecutionRecord(
                            symbol=order.symbol,
                            side=order.side,
                            submitted_qty=0.0,
                            status="skipped",
                            reason=reason,
                        )
                    )
                    continue

                submitted_qty = float(qty_str)
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
                status = "failed" if failed else parse_order_status(result_text)
                fills.append(
                    ExecutionRecord(
                        symbol=order.symbol,
                        side=order.side,
                        submitted_qty=submitted_qty,
                        status=status,
                        limit_price=limit_px if use_limit else None,
                    )
                )
                log.line(
                    f"{order.side.upper()} {order.symbol} x{qty_str} → {status}"
                )
                log.line(f"  {summary}")
                if use_limit:
                    ref = last if last > 0 else (ask or bid)
                    if ref > 0:
                        slip = (limit_px / ref - 1.0) * 100.0
                        log.line(
                            f"  Limit ${limit_px:.2f} vs last ${last:.2f} "
                            f"NBBO {bid:.2f}/{ask:.2f} ({slip:+.2f}%)"
                        )
                else:
                    log.line("  Market order (no usable NBBO; last-close limit skipped)")
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
                fills.append(
                    ExecutionRecord(
                        symbol=order.symbol,
                        side=order.side,
                        submitted_qty=0.0,
                        status="error",
                        reason=str(exc),
                    )
                )
        return fills


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
    else:
        log.line("Candidates: (none)")


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
    return f"[{tickers}] {event.summary} (importance {event.importance}/5)"


def _log_decision(decision: PortfolioDecision, log: CycleLog) -> None:
    log.section("Decision")
    if decision.is_trade():
        for order in decision.orders:
            log.line(f"Decision: {order.side.upper()} {order.symbol} x{order.quantity:g}")
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
