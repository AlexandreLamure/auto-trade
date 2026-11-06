"""Vote tally helpers for committee deliberation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, median

from agent.decision import PersonaProposal


@dataclass
class SymbolTally:
    symbol: str
    buy_votes: int = 0
    sell_votes: int = 0
    hold_votes: int = 0
    buy_qty: list[float] = field(default_factory=list)
    sell_qty: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)

    @property
    def majority_side(self) -> str | None:
        if self.buy_votes > self.sell_votes and self.buy_votes > 0:
            return "buy"
        if self.sell_votes > self.buy_votes and self.sell_votes > 0:
            return "sell"
        return None

    @property
    def majority_count(self) -> int:
        return max(self.buy_votes, self.sell_votes)

    @property
    def mean_confidence(self) -> float:
        return mean(self.confidences) if self.confidences else 0.0

    @property
    def median_qty(self) -> float:
        qty = self.buy_qty if self.majority_side == "buy" else self.sell_qty
        return float(median(qty)) if qty else 0.0


def tally_proposals(proposals: list[PersonaProposal]) -> dict[str, SymbolTally]:
    """Count directional votes from trader proposals."""
    tallies: dict[str, SymbolTally] = {}
    n_traders = max(1, len(proposals))
    voted: dict[str, set[str]] = defaultdict(set)

    for proposal in proposals:
        if not proposal.orders:
            continue
        for order in proposal.orders:
            symbol = order.symbol.upper()
            row = tallies.setdefault(symbol, SymbolTally(symbol=symbol))
            if proposal.persona in voted[symbol]:
                continue
            voted[symbol].add(proposal.persona)
            row.personas.append(proposal.persona)
            row.confidences.append(proposal.confidence)
            if order.side == "buy":
                row.buy_votes += 1
                row.buy_qty.append(order.quantity)
            elif order.side == "sell":
                row.sell_votes += 1
                row.sell_qty.append(order.quantity)

    for proposal in proposals:
        for symbol, row in tallies.items():
            if proposal.persona in voted[symbol]:
                continue
            row.hold_votes += 1

    return tallies


def format_vote_tally(tallies: dict[str, SymbolTally]) -> str:
    if not tallies:
        return "_No directional votes._"
    lines = ["| Symbol | Buy | Sell | Hold | Mean conf | Median qty |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for symbol in sorted(tallies):
        row = tallies[symbol]
        lines.append(
            f"| {symbol} | {row.buy_votes} | {row.sell_votes} | {row.hold_votes} | "
            f"{row.mean_confidence:.2f} | {row.median_qty:g} |"
        )
    return "\n".join(lines)


def has_disagreement(proposals: list[PersonaProposal]) -> bool:
    """True when traders disagree on side for any symbol, or mix HOLD with trades."""
    tallies = tally_proposals(proposals)
    if not tallies:
        return False
    holding = sum(1 for p in proposals if not p.orders)
    trading = sum(1 for p in proposals if p.orders)
    if holding and trading:
        return True
    return any(row.buy_votes > 0 and row.sell_votes > 0 for row in tallies.values())
