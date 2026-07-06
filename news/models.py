"""Shared data models for the news/signal pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from news.weights import UNKNOWN_SOURCE_WEIGHT


@dataclass
class Signal:
    url: str
    title: str
    source: str
    published_at: datetime | None
    snippet: str
    tickers_hint: list[str] = field(default_factory=list)
    weight: float = UNKNOWN_SOURCE_WEIGHT
