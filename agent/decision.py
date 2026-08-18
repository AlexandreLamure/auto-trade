"""
Portfolio and persona decision parsers.

Committee personas output JSON inside fenced code blocks; these parsers
extract and validate that structured output.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from util.json_parse import extract_json_block

logger = logging.getLogger(__name__)

_PARSE_FAILURE_REASON = "Could not parse a valid decision from LLM output."

_THINK_PATTERNS = (
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
)


@dataclass
class TradeOrder:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    rationale: str
    confidence: float = 0.0


@dataclass
class PortfolioDecision:
    orders: list[TradeOrder]
    consensus_summary: str
    dissent: str | None = None

    @property
    def parsed_ok(self) -> bool:
        return self.consensus_summary != _PARSE_FAILURE_REASON

    def is_trade(self) -> bool:
        return bool(self.orders)


@dataclass
class PersonaProposal:
    persona: str
    stance: str
    orders: list[TradeOrder]
    confidence: float
    key_points: list[str]
    commentary: str = ""

    @property
    def parsed_ok(self) -> bool:
        return bool(self.commentary or self.key_points or self.orders or self.stance)


def _hold_fallback(persona: str, commentary: str) -> PersonaProposal:
    return PersonaProposal(
        persona=persona,
        stance="HOLD",
        orders=[],
        confidence=0.0,
        key_points=[],
        commentary=commentary,
    )


def order_to_dict(order: TradeOrder) -> dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "qty": order.quantity,
        "rationale": order.rationale,
    }


def proposal_to_dict(proposal: PersonaProposal) -> dict[str, Any]:
    return {
        "persona": proposal.persona,
        "stance": proposal.stance,
        "confidence": proposal.confidence,
        "key_points": proposal.key_points,
        "commentary": proposal.commentary,
        "orders": [order_to_dict(o) for o in proposal.orders],
    }


def _fallback_portfolio_hold(reason: str = _PARSE_FAILURE_REASON) -> PortfolioDecision:
    return PortfolioDecision(orders=[], consensus_summary=reason, dissent=None)


def parse_portfolio_decision(text: str, *, extra_text: str | None = None) -> PortfolioDecision:
    """Extract and validate a PortfolioDecision from LLM output."""
    for candidate in (text, extra_text):
        if not candidate:
            continue
        decision = _parse_portfolio_decision_text(candidate)
        if decision.parsed_ok:
            return decision

    logger.warning("No portfolio decision JSON found; defaulting to HOLD")
    return _fallback_portfolio_hold()


def parse_persona_proposal(text: str, *, persona: str = "", extra_text: str | None = None) -> PersonaProposal:
    """Extract a trader persona proposal from LLM output."""
    for candidate in (text, extra_text):
        if not candidate:
            continue
        proposal = _parse_persona_proposal_text(candidate, persona=persona)
        if proposal.parsed_ok:
            return proposal

    return PersonaProposal(
        persona=persona,
        stance="HOLD",
        orders=[],
        confidence=0.0,
        key_points=[],
        commentary=text[:500] if text else _PARSE_FAILURE_REASON,
    )


def _parse_portfolio_decision_text(text: str) -> PortfolioDecision:
    clean = _strip_thinking(text)
    raw_json = _extract_json_str(clean, marker_keys=("orders", "consensus_summary"))
    if raw_json is None:
        return _fallback_portfolio_hold()

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error in portfolio decision: %s", exc)
        return _fallback_portfolio_hold()

    if not isinstance(data, dict):
        return _fallback_portfolio_hold()

    return _validate_portfolio(data)


def _parse_persona_proposal_text(text: str, *, persona: str) -> PersonaProposal:
    clean = _strip_thinking(text)
    raw_json = _extract_json_str(clean, marker_keys=("stance", "orders", "confidence"))
    if raw_json is None:
        return _hold_fallback(persona, clean[:800])

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return _hold_fallback(persona, clean[:800])

    if not isinstance(data, dict):
        return _hold_fallback(persona, clean[:800])

    orders = _parse_order_list(data.get("orders") or [])
    key_points_raw = data.get("key_points") or []
    key_points = [str(p) for p in key_points_raw] if isinstance(key_points_raw, list) else []
    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    return PersonaProposal(
        persona=persona,
        stance=str(data.get("stance", "HOLD")).upper(),
        orders=orders,
        confidence=confidence,
        key_points=key_points,
        commentary=str(data.get("commentary", "") or data.get("rebuttal", "") or "").strip(),
    )


def _strip_thinking(text: str) -> str:
    clean = text
    for pattern in _THINK_PATTERNS:
        clean = pattern.sub("", clean)
    return clean.strip()


def _parse_order_list(raw_orders: Any) -> list[TradeOrder]:
    if not isinstance(raw_orders, list):
        return []
    orders: list[TradeOrder] = []
    for item in raw_orders:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        side_raw = str(item.get("side", "")).lower().strip()
        if side_raw not in ("buy", "sell"):
            if str(item.get("action", "")).upper() == "SELL":
                side_raw = "sell"
            elif str(item.get("action", "")).upper() == "BUY":
                side_raw = "buy"
            else:
                continue
        try:
            qty = float(item.get("qty", item.get("quantity", 0)))
        except (TypeError, ValueError):
            continue
        if not symbol or qty <= 0:
            continue
        rationale = str(item.get("rationale", item.get("reasoning", ""))).strip()
        orders.append(TradeOrder(symbol=symbol, side=side_raw, quantity=qty, rationale=rationale))
    return orders


def _validate_portfolio(data: dict) -> PortfolioDecision:
    orders = _parse_order_list(data.get("orders") or [])
    summary = str(
        data.get("consensus_summary", data.get("reasoning", data.get("summary", "")))
    ).strip()
    dissent_raw = data.get("dissent")
    dissent = str(dissent_raw).strip() if dissent_raw else None
    if not summary:
        summary = _PARSE_FAILURE_REASON if not orders else "Committee consensus reached."
    return PortfolioDecision(orders=orders, consensus_summary=summary, dissent=dissent)


def _extract_json_str(text: str, marker_keys: tuple[str, ...] = ("action",)) -> str | None:
    block = extract_json_block(text)
    if block:
        return block
    return _extract_balanced_json_object(text, marker_keys=marker_keys)


def _extract_balanced_json_object(
    text: str,
    *,
    marker_keys: tuple[str, ...] = ("action",),
) -> str | None:
    """Find the first {...} object that contains a known marker key."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    lower = candidate.lower()
                    if any(f'"{key}"' in lower for key in marker_keys):
                        return candidate
                    break
        start = text.find("{", start + 1)
    return None
