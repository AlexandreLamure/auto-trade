"""
AgentOrchestrator – committee trading cycle.

Each call to run_cycle():
  1. Deep programmatic research (MCP tools)
  2. Multi-persona deliberation → PortfolioDecision
  3. Risk validation and multi-order execution
  4. Logs to runs.jsonl + logs/discussions/
"""

from __future__ import annotations

import io
import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import Settings
from servers.manager import MCPManager
from agent.llm_client import OllamaClient
from agent.decision import PortfolioDecision, TradeOrder
from agent.deliberation import run_committee
from agent.research import run_deep_research
from agent.risk import build_validation_context, validate_orders

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
RUNS_LOG = LOGS_DIR / "runs.jsonl"
DISCUSSIONS_DIR = LOGS_DIR / "discussions"


class AgentOrchestrator:
    def __init__(self, settings: Settings, verbose: bool = False) -> None:
        self._settings = settings
        self._verbose = verbose or settings.verbose
        self._llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            enable_thinking=settings.enable_thinking,
        )
        LOGS_DIR.mkdir(exist_ok=True)
        DISCUSSIONS_DIR.mkdir(exist_ok=True)
        _validate_settings(settings)

    async def run_cycle(self) -> None:
        """Execute one full analysis-and-trade cycle."""
        started_at = datetime.now(timezone.utc)
        cycle_id = started_at.strftime("%Y%m%dT%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
        logger.info("=== Cycle %s started at %s ===", cycle_id, started_at.isoformat())

        run_log: dict[str, Any] = {
            "cycle_id": cycle_id,
            "timestamp": started_at.isoformat(),
            "mode": "committee",
            "model": self._settings.ollama_model,
            "tool_calls": [],
            "decision": None,
            "execution": None,
            "error": None,
        }

        try:
            async with MCPManager(self._settings) as manager:
                await self._run_committee_cycle(manager, cycle_id, run_log)

        except Exception as exc:  # noqa: BLE001
            logger.error("Cycle failed: %s", exc, exc_info=True)
            run_log["error"] = traceback.format_exc()
        finally:
            self._append_log(run_log)
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            logger.info("=== Cycle completed in %.1fs ===", elapsed)

    async def _run_committee_cycle(
        self,
        manager: MCPManager,
        cycle_id: str,
        run_log: dict[str, Any],
    ) -> None:
        logger.info("Phase 1: deep research")
        brief = await run_deep_research(manager, self._settings)
        run_log["tool_calls"] = brief.tool_call_log
        run_log["portfolio_snapshot"] = {
            "equity": brief.portfolio_equity,
            "cash": brief.cash_available,
            "month_pnl_pct": brief.month_pnl_pct,
            "held_symbols": brief.held_symbols,
            "candidate_symbols": brief.candidate_symbols,
        }
        run_log["research_summary"] = brief.summary_markdown[:4000]

        logger.info("Phase 2: committee deliberation")
        portfolio_decision, transcript = await run_committee(
            self._llm,
            brief,
            self._settings,
            cycle_id=cycle_id,
            verbose=self._verbose,
            print_section=_print_section if self._verbose else None,
        )
        run_log["deliberation_id"] = cycle_id
        self._save_discussion_transcript(transcript)

        run_log["decision"] = {
            "consensus_summary": portfolio_decision.consensus_summary,
            "dissent": portfolio_decision.dissent,
            "orders": [
                {
                    "symbol": o.symbol,
                    "side": o.side,
                    "quantity": o.quantity,
                    "rationale": o.rationale,
                }
                for o in portfolio_decision.orders
            ],
        }
        _log_portfolio_decision(portfolio_decision)

        logger.info("Phase 3: execution")
        ctx = build_validation_context(
            self._settings,
            cash_available=brief.cash_available,
            positions_json=brief.raw_sections.get("positions", ""),
            latest_prices=brief.latest_prices,
        )
        approved, rejections = validate_orders(portfolio_decision.orders, ctx)
        if rejections:
            logger.warning("Order validation notes: %s", "; ".join(rejections))
            run_log["validation_rejections"] = rejections

        if approved:
            executions = await self._execute_portfolio(manager, approved)
            run_log["execution"] = executions
        else:
            run_log["execution"] = {"status": "hold", "orders": []}

    async def _execute_portfolio(
        self,
        manager: MCPManager,
        orders: list[TradeOrder],
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for order in orders:
            logger.info(
                "Executing %s %s x%.0f via Alpaca",
                order.side.upper(),
                order.symbol,
                order.quantity,
            )
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
                logger.info("Order result: %s", result_text[:500])
                try:
                    detail = json.loads(result_text)
                except json.JSONDecodeError:
                    detail = {"raw": result_text}
                results.append(
                    {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                        "status": "submitted",
                        "detail": detail,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Trade execution failed for %s: %s", order.symbol, exc)
                results.append(
                    {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                        "status": "error",
                        "detail": str(exc),
                    }
                )
        return {"status": "submitted", "orders": results}

    def _save_discussion_transcript(self, transcript: Any) -> None:
        path = DISCUSSIONS_DIR / f"{transcript.cycle_id}.json"
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(transcript.to_dict(), fh, ensure_ascii=False, indent=2, default=str)
            logger.info("Discussion transcript saved: %s", path)
        except OSError as exc:
            logger.error("Failed to write discussion log: %s", exc)

    def _append_log(self, record: dict[str, Any]) -> None:
        try:
            with RUNS_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.error("Failed to write run log: %s", exc)


def _log_portfolio_decision(decision: PortfolioDecision) -> None:
    if decision.is_trade():
        for order in decision.orders:
            logger.info("Decision: %s %s x%.0f", order.side.upper(), order.symbol, order.quantity)
    else:
        logger.info("Decision: HOLD – no orders")
    if decision.consensus_summary:
        logger.info("Consensus: %s", decision.consensus_summary)
    if decision.dissent:
        logger.info("Dissent: %s", decision.dissent)


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


_DIVIDER = "-" * 72


def _print_section(title: str, content: str | None) -> None:
    """Print a labelled section to stdout (only used in verbose mode)."""
    if not content:
        return
    import sys

    out = sys.stdout
    try:
        out.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation):
        pass
    text = f"\n{_DIVIDER}\n  {title}\n{_DIVIDER}\n{content.strip()}\n{_DIVIDER}\n"
    try:
        out.write(text)
        out.flush()
    except UnicodeEncodeError:
        out.write(text.encode("ascii", errors="replace").decode("ascii"))
        out.flush()
