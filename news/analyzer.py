"""Deduplicate articles, group into events, and LLM-enrich."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from agent.llm_client import OllamaClient
from config.settings import Settings
from news.models import RawArticle
from news.sources.base import (
    extract_tickers,
    normalize_title,
    title_similarity,
    url_hash,
)
from store import (
    article_exists,
    create_event,
    find_candidate_events,
    get_event_articles,
    insert_article,
    update_event,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

EVENT_MATCH_THRESHOLD = 0.4
HEADLINE_MATCH_THRESHOLD = 0.85
MAX_ENRICHMENTS_PER_CYCLE = 25

ENRICH_SYSTEM_PROMPT = """\
You are a financial news analyst. Given multiple articles about the same market event,
produce a concise structured summary for traders.

Respond with ONLY a fenced JSON block:
```json
{
  "summary": "2-4 sentence summary of the event and market impact",
  "event_type": "earnings|macro|regulatory|m_and_a|product|guidance|analyst|other",
  "sentiment": "bullish|bearish|neutral",
  "importance": 1,
  "confidence": 0.8,
  "tickers": ["AAPL"],
  "companies": ["Apple Inc"]
}
```

importance: 1=minor, 5=major market-moving. confidence: 0.0-1.0 how certain the analysis is."""


@dataclass
class EnrichedEvent:
    summary: str
    event_type: str
    sentiment: str
    importance: int
    confidence: float
    tickers: list[str]
    companies: list[str]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_enrichment(text: str) -> EnrichedEvent | None:
    match = _JSON_BLOCK_RE.search(text)
    raw = match.group(1) if match else text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    importance = data.get("importance", 1)
    confidence = data.get("confidence", 0.5)
    try:
        importance = max(1, min(5, int(importance)))
    except (TypeError, ValueError):
        importance = 1
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5

    tickers = data.get("tickers") or []
    companies = data.get("companies") or []
    if not isinstance(tickers, list):
        tickers = []
    if not isinstance(companies, list):
        companies = []

    return EnrichedEvent(
        summary=str(data.get("summary") or "").strip(),
        event_type=str(data.get("event_type") or "other").strip(),
        sentiment=str(data.get("sentiment") or "neutral").strip(),
        importance=importance,
        confidence=confidence,
        tickers=[str(t).upper() for t in tickers if t],
        companies=[str(c) for c in companies if c],
    )


def _match_event(
    article: RawArticle,
    candidates: list,
) -> str | None:
    norm = normalize_title(article.title)
    for event in candidates:
        if title_similarity(article.title, event.summary) >= EVENT_MATCH_THRESHOLD:
            return event.id
        # Compare against event summary headline-like start
        if title_similarity(article.title, event.summary[:120]) >= EVENT_MATCH_THRESHOLD:
            return event.id
    for event in candidates:
        if normalize_title(event.summary[:80]) == norm[:80]:
            return event.id
    return None


def _article_tickers(article: RawArticle, watchlist: list[str]) -> list[str]:
    hints = list(article.tickers_hint)
    extracted = extract_tickers(f"{article.title} {article.snippet}", known=watchlist)
    return list(dict.fromkeys(hints + extracted))


async def _enrich_event(
    llm: OllamaClient,
    articles: list[RawArticle],
) -> EnrichedEvent:
    lines = []
    for i, art in enumerate(articles, 1):
        lines.append(f"Article {i}:")
        lines.append(f"  Title: {art.title}")
        lines.append(f"  Source: {art.source}")
        lines.append(f"  Snippet: {art.snippet[:300]}")
        lines.append("")

    user_prompt = "Analyze these related articles:\n\n" + "\n".join(lines)

    response = await llm.chat(
        [
            {"role": "system", "content": ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = response.content or ""
    enriched = _parse_enrichment(content)
    if enriched and enriched.summary:
        return enriched

    # Fallback without LLM parse
    title = articles[0].title if articles else "Unknown event"
    tickers = _article_tickers(articles[0], []) if articles else []
    return EnrichedEvent(
        summary=title,
        event_type="other",
        sentiment="neutral",
        importance=2,
        confidence=0.3,
        tickers=tickers,
        companies=[],
    )


async def process_articles(
    settings: Settings,
    articles: list[RawArticle],
    llm: OllamaClient,
) -> dict[str, int]:
    """Ingest new articles: dedupe, group, enrich, persist. Returns stats."""
    db_path = settings.event_store_path
    watchlist = [
        s.strip().upper()
        for s in settings.watchlist_tickers.split(",")
        if s.strip()
    ]

    stats = {"fetched": len(articles), "new": 0, "skipped": 0, "updated_events": 0}

    events_to_enrich: dict[str, list[RawArticle]] = {}

    for article in articles:
        h = url_hash(article.url)
        if article_exists(db_path, h):
            stats["skipped"] += 1
            continue

        tickers = _article_tickers(article, watchlist)
        candidates = find_candidate_events(db_path, tickers, since_hours=48)

        event_id = _match_event(article, candidates)
        if event_id is None and candidates:
            # Try broader match on any candidate with overlapping tickers
            for event in candidates:
                if title_similarity(article.title, event.summary) >= HEADLINE_MATCH_THRESHOLD:
                    event_id = event.id
                    break

        if event_id is None:
            event_id = create_event(
                db_path,
                summary=article.title,
                tickers=tickers,
                first_seen_at=article.published_at or _utcnow(),
                last_seen_at=article.published_at or _utcnow(),
            )

        insert_article(
            db_path,
            url=article.url,
            url_hash=h,
            title=article.title,
            source=article.source,
            snippet=article.snippet,
            event_id=event_id,
            published_at=article.published_at,
        )
        stats["new"] += 1

        stored = get_event_articles(db_path, event_id)
        # Reconstruct RawArticle list for enrichment from DB + current
        raw_for_event = [
            RawArticle(
                url=a.url,
                title=a.title,
                source=a.source,
                published_at=a.published_at,
                snippet=a.snippet,
            )
            for a in stored
        ]
        events_to_enrich[event_id] = raw_for_event

    enriched_count = 0
    for event_id, event_articles in events_to_enrich.items():
        if enriched_count >= MAX_ENRICHMENTS_PER_CYCLE:
            logger.info(
                "Enrichment cap reached (%d); remaining events keep stub summaries",
                MAX_ENRICHMENTS_PER_CYCLE,
            )
            break
        enriched = await _enrich_event(llm, event_articles)
        last_seen = max(
            (a.published_at or _utcnow() for a in event_articles),
            default=_utcnow(),
        )
        update_event(
            db_path,
            event_id,
            summary=enriched.summary,
            event_type=enriched.event_type,
            sentiment=enriched.sentiment,
            importance=enriched.importance,
            confidence=enriched.confidence,
            tickers=enriched.tickers,
            companies=enriched.companies,
            article_count=len(event_articles),
            last_seen_at=last_seen,
        )
        stats["updated_events"] += 1
        enriched_count += 1

    return stats
