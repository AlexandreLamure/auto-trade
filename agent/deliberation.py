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
from dataclasses import dataclass

from config.settings import Settings
from agent.decision import (
    PersonaProposal,
    PortfolioDecision,
    parse_persona_proposal,
    parse_portfolio_decision,
    proposal_to_dict,
)
from agent.personas import (
    CHAIR_JSON_SCHEMA,
    DEBATE_JSON_SCHEMA,
    HORIZON_DAYS,
    PROPOSAL_JSON_SCHEMA,
    Persona,
    build_chair_persona,
    build_trader_personas,
)
from agent.consensus import apply_consensus_gate
from agent.research import ResearchBrief
from util.cycle_log import CycleLog
from util.llm_client import OllamaClient
from util.text import truncate_text

logger = logging.getLogger(__name__)


@dataclass
class DeliberationTurn:
    persona: str
    persona_name: str
    round: int
    content: str
    thinking: str | None = None
    proposal: PersonaProposal | None = None


def _sizing_hint(brief: ResearchBrief, max_position_pct: float) -> str:
    equity = brief.portfolio_equity if brief.portfolio_equity > 0 else brief.cash_available
    max_dollars = equity * max_position_pct
    return (
        f"Equity: ${equity:,.2f} | Cash: ${brief.cash_available:,.2f} | "
        f"Max per name: ${max_dollars:,.2f} ({max_position_pct:.0%} of equity). "
        "HOLD is valid. Size from conviction, not to fill the cap."
    )


def _build_round1_user_prompt(
    brief: ResearchBrief, *, max_position_pct: float
) -> str:
    return f"""\
Analyse the research brief below and submit your independent trading proposal.
HOLD is acceptable when you do not see a 30-day edge.

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
You may HOLD. Do not invent orders you cannot defend from the brief.

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
- Execute a symbol only when 2+ traders agree on direction with a real thesis.
- Up to {max_orders} orders. HOLD (empty list) when agreement or catalysts are missing.
- Net conflicting orders on the same symbol.
- Size buys up to ~{max_position_pct:.0%} of equity per name ({_sizing_hint(brief, max_position_pct)}).
- consensus_summary must explain the {HORIZON_DAYS}-day PnL strategy, including why cash is held if it is.

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

    proposal = None
    if persona.id != "chair":
        proposal = parse_persona_proposal(
            content, persona=persona.id, extra_text=response.thinking
        )

    return DeliberationTurn(
        persona=persona.id,
        persona_name=persona.name,
        round=round_num,
        content=content,
        thinking=response.thinking,
        proposal=proposal,
    )


async def run_committee(
    llm: OllamaClient,
    brief: ResearchBrief,
    settings: Settings,
    *,
    cycle_log: CycleLog | None = None,
) -> PortfolioDecision:
    """Run multi-round deliberation and return consensus."""
    started = time.monotonic()
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

    rounds: list[DeliberationTurn] = []

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
    rounds.extend(round1_turns)

    round1_proposals = [
        t.proposal
        or parse_persona_proposal(t.content, persona=t.persona, extra_text=t.thinking)
        for t in round1_turns
    ]

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
            rounds.append(turn)

    final_proposals = list(round1_proposals)
    round2_turns = [t for t in rounds if t.round == 2]
    if round2_turns:
        final_proposals = [
            t.proposal
            or parse_persona_proposal(t.content, persona=t.persona, extra_text=t.thinking)
            for t in round2_turns
        ]

    chair_prompt = _build_chair_user_prompt(
        brief,
        rounds,
        max_position_pct=settings.max_position_pct,
        max_orders=settings.max_orders_per_cycle,
    )
    chair_turn = await _call_persona(
        llm, chair, chair_prompt, round_num=3, cycle_log=cycle_log
    )

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
        decision = parse_portfolio_decision(
            chair_turn_retry.content, extra_text=chair_turn_retry.thinking
        )

    return apply_consensus_gate(decision, final_proposals, brief, settings)
