"""
Multi-round investment committee deliberation.

Phase 2 of the committee pipeline: traders analyse a shared ResearchBrief,
debate, then the chair produces a PortfolioDecision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config.settings import Settings
from agent.decision import (
    PersonaProposal,
    PortfolioDecision,
    build_portfolio_snapshot,
    parse_persona_proposal,
    parse_portfolio_decision,
    portfolio_to_dict,
    proposal_to_dict,
)
from agent.llm_client import OllamaClient
from agent.personas import (
    CHAIR_JSON_SCHEMA,
    DEBATE_JSON_SCHEMA,
    HORIZON_DAYS,
    PROPOSAL_JSON_SCHEMA,
    Persona,
    build_chair_persona,
    build_trader_personas,
)
from agent.research import ResearchBrief
from agent.workflow import truncate_text
from util.cycle_log import CycleLog

logger = logging.getLogger(__name__)


@dataclass
class DeliberationTurn:
    persona: str
    persona_name: str
    round: int
    content: str
    thinking: str | None = None
    parsed_proposal: dict[str, Any] | None = None


@dataclass
class DeliberationTranscript:
    cycle_id: str
    started_at: str
    duration_seconds: float
    research_summary: str
    portfolio_snapshot: dict[str, Any]
    rounds: list[DeliberationTurn] = field(default_factory=list)
    consensus: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "research_summary": self.research_summary,
            "portfolio_snapshot": self.portfolio_snapshot,
            "rounds": [
                {
                    "persona": t.persona,
                    "persona_name": t.persona_name,
                    "round": t.round,
                    "content": t.content,
                    "thinking": t.thinking,
                    "parsed_proposal": t.parsed_proposal,
                }
                for t in self.rounds
            ],
            "consensus": self.consensus,
        }


def _proposal_from_turn(turn: DeliberationTurn) -> PersonaProposal:
    if turn.parsed_proposal:
        data = turn.parsed_proposal
        from agent.decision import _parse_order_list

        return PersonaProposal(
            persona=data.get("persona", turn.persona),
            stance=str(data.get("stance", "HOLD")).upper(),
            orders=_parse_order_list(data.get("orders") or []),
            confidence=float(data.get("confidence", 0)),
            key_points=list(data.get("key_points") or []),
            commentary=str(data.get("commentary", "") or ""),
        )
    return parse_persona_proposal(turn.content, persona=turn.persona, extra_text=turn.thinking)


def _sizing_hint(brief: ResearchBrief, max_position_pct: float) -> str:
    max_dollars = brief.cash_available * max_position_pct
    return (
        f"Cash: ${brief.cash_available:,.2f} | Max per buy: ${max_dollars:,.2f} "
        f"({max_position_pct:.0%} of cash). Size qty to use most of that budget on high-conviction buys."
    )


def _build_round1_user_prompt(
    brief: ResearchBrief, *, max_position_pct: float
) -> str:
    return f"""\
Analyse the research brief below and submit your independent trading proposal.
Propose concrete orders — avoid defaulting to HOLD unless you truly see no edge.

{PROPOSAL_JSON_SCHEMA}

## Sizing
{_sizing_hint(brief, max_position_pct)}

## Research brief
{brief.to_prompt_context()}"""


def _build_debate_user_prompt(
    brief: ResearchBrief,
    proposals: list[PersonaProposal],
    *,
    max_position_pct: float,
) -> str:
    proposal_text = json.dumps([proposal_to_dict(p) for p in proposals], indent=2)
    return f"""\
Review the other traders' Round 1 proposals. Agree or disagree, then submit a revised view.
Do not retreat to HOLD just because others disagree — defend your thesis with sized orders or propose alternatives.

{DEBATE_JSON_SCHEMA}

## Sizing
{_sizing_hint(brief, max_position_pct)}

## Research brief (summary)
{brief.debate_context()}

## Round 1 proposals
{proposal_text}"""


def _build_chair_user_prompt(
    brief: ResearchBrief,
    transcript: list[DeliberationTurn],
    *,
    max_position_pct: float,
    max_orders: int,
) -> str:
    debate_lines = []
    for turn in transcript:
        debate_lines.append(
            f"### Round {turn.round} — {turn.persona_name} ({turn.persona})\n{turn.content}"
        )
    debate_text = "\n\n".join(debate_lines)
    return f"""\
As portfolio chair, synthesise the committee debate into a final consensus.

{CHAIR_JSON_SCHEMA}

Rules:
- **Default to executing** the best ideas from debate — do not let risk aversion deadlock the committee.
- Up to {max_orders} orders; prefer 1–{max_orders} actionable orders when any trader had a credible thesis.
- HOLD (empty list) only when no trader presented a tradeable idea or conditions are clearly adverse.
- Net conflicting orders on the same symbol.
- Size buys to use ~{max_position_pct:.0%} of cash per order ({_sizing_hint(brief, max_position_pct)}).
- consensus_summary must explain the {HORIZON_DAYS}-day PnL strategy.

## Research brief
{brief.to_prompt_context()}

## Committee transcript
{debate_text}"""


async def _call_persona(
    llm: OllamaClient,
    persona: Persona,
    user_prompt: str,
    *,
    round_num: int,
    cycle_log: CycleLog | None,
) -> DeliberationTurn:
    messages = [
        {"role": "system", "content": persona.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = await llm.chat(messages)
    content = response.content or ""

    if cycle_log is not None:
        cycle_log.line(f"[{persona.name} · Round {round_num}]")
        if response.thinking:
            cycle_log.line(truncate_text(response.thinking, 2000))
        cycle_log.line(truncate_text(content, 4000))
        cycle_log.line("")

    if persona.id == "chair":
        parsed = parse_portfolio_decision(content, extra_text=response.thinking)
        parsed_dict = portfolio_to_dict(parsed)
    else:
        proposal = parse_persona_proposal(content, persona=persona.id, extra_text=response.thinking)
        parsed_dict = proposal_to_dict(proposal)

    return DeliberationTurn(
        persona=persona.id,
        persona_name=persona.name,
        round=round_num,
        content=content,
        thinking=response.thinking,
        parsed_proposal=parsed_dict,
    )


async def run_committee(
    llm: OllamaClient,
    brief: ResearchBrief,
    settings: Settings,
    *,
    cycle_id: str,
    cycle_log: CycleLog | None = None,
) -> tuple[PortfolioDecision, DeliberationTranscript]:
    """Run multi-round deliberation and return consensus + transcript."""
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    deadline = started + settings.max_cycle_seconds

    traders = build_trader_personas(
        HORIZON_DAYS,
        settings.max_position_pct,
        settings.max_orders_per_cycle,
    )
    chair = build_chair_persona(
        HORIZON_DAYS,
        settings.max_position_pct,
        settings.max_orders_per_cycle,
    )

    transcript = DeliberationTranscript(
        cycle_id=cycle_id,
        started_at=started_at,
        duration_seconds=0.0,
        research_summary=truncate_text(brief.summary_markdown, 4000),
        portfolio_snapshot=build_portfolio_snapshot(brief),
    )

    round1_prompt = _build_round1_user_prompt(
        brief, max_position_pct=settings.max_position_pct
    )
    round1_tasks = [
        _call_persona(
            llm, persona, round1_prompt, round_num=1, cycle_log=cycle_log
        )
        for persona in traders
    ]
    round1_turns = await asyncio.gather(*round1_tasks)
    transcript.rounds.extend(round1_turns)

    round1_proposals = [_proposal_from_turn(t) for t in round1_turns]

    if settings.enable_debate_round and time.monotonic() < deadline:
        debate_prompt = _build_debate_user_prompt(
            brief, round1_proposals, max_position_pct=settings.max_position_pct
        )
        for persona in traders:
            if time.monotonic() >= deadline:
                logger.warning("Cycle time budget exceeded; skipping remaining debate")
                break
            turn = await _call_persona(
                llm,
                persona,
                debate_prompt,
                round_num=2,
                cycle_log=cycle_log,
            )
            transcript.rounds.append(turn)

    chair_prompt = _build_chair_user_prompt(
        brief,
        transcript.rounds,
        max_position_pct=settings.max_position_pct,
        max_orders=settings.max_orders_per_cycle,
    )
    chair_turn = await _call_persona(
        llm, chair, chair_prompt, round_num=3, cycle_log=cycle_log
    )
    transcript.rounds.append(chair_turn)

    decision = parse_portfolio_decision(chair_turn.content, extra_text=chair_turn.thinking)
    if not decision.parsed_ok:
        logger.warning("Chair parse failed; retrying once")
        chair_turn_retry = await _call_persona(
            llm,
            chair,
            chair_prompt + "\n\nYour previous response was not valid JSON. Output ONLY the JSON block.",
            round_num=3,
            cycle_log=cycle_log,
        )
        transcript.rounds.append(chair_turn_retry)
        decision = parse_portfolio_decision(
            chair_turn_retry.content, extra_text=chair_turn_retry.thinking
        )

    transcript.consensus = portfolio_to_dict(decision)
    transcript.duration_seconds = time.monotonic() - started
    return decision, transcript
