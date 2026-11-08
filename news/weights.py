"""Source weight resolution."""

from __future__ import annotations

UNKNOWN_SOURCE_WEIGHT = 0.5

_SOURCE_PREFIX_MAP: dict[str, str] = {
    "sec_edgar": "sec_edgar",
    "ir": "company_ir",
    "rss": "rss",
    "api": "api",
    "reddit": "reddit",
    "stocktwits": "stocktwits",
}


def resolve_weight(source: str, weights: dict[str, float]) -> float:
    """Map a signal source label to its confidence weight."""
    prefix = source.split(":")[0] if ":" in source else source
    weight_key = _SOURCE_PREFIX_MAP.get(prefix, prefix)
    return weights.get(weight_key, UNKNOWN_SOURCE_WEIGHT)

