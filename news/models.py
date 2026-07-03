"""Shared data models for the news pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawArticle:
    url: str
    title: str
    source: str
    published_at: datetime | None
    snippet: str
    tickers_hint: list[str] = field(default_factory=list)
