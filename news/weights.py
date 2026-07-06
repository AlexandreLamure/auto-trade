"""Source weight resolution and weighted event scoring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from news.models import Signal

UNKNOWN_SOURCE_WEIGHT = 0.5

_SOURCE_PREFIX_MAP: dict[str, str] = {
    "sec_edgar": "sec_edgar",
    "ir": "company_ir",
    "rss": "rss",
    "api": "api",
    "polymarket": "polymarket",
    "reddit": "reddit",
    "stocktwits": "stocktwits",
    "google_trends": "google_trends",
}


def parse_source_weights(raw: str) -> dict[str, float]:
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in data.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def resolve_weight(source: str, weights: dict[str, float]) -> float:
    """Map a signal source label to its confidence weight."""
    prefix = source.split(":")[0] if ":" in source else source
    weight_key = _SOURCE_PREFIX_MAP.get(prefix, prefix)
    return weights.get(weight_key, UNKNOWN_SOURCE_WEIGHT)


def apply_weighted_scores(
    signals: list[Signal],
    *,
    importance: int,
    confidence: float,
) -> tuple[int, float]:
    """Adjust LLM-derived importance and confidence using signal source weights."""
    if not signals:
        return importance, confidence

    avg_w = sum(s.weight for s in signals) / len(signals)
    max_w = max(s.weight for s in signals)
    high_w = sum(1 for s in signals if s.weight >= 0.9)

    adj_confidence = min(1.0, confidence * (0.7 + 0.3 * avg_w) + 0.05 * high_w)
    adj_importance = min(
        5,
        max(1, round(importance * (0.6 + 0.4 * max_w) + 0.1 * (len(signals) - 1))),
    )
    return adj_importance, adj_confidence
