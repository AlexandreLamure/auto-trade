"""Deduplicate signals, group into events, and LLM-enrich."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from config.settings import Settings
from news.models import Signal
from news.sources.base import (
    extract_tickers,
    normalize_title,
    title_similarity,
    url_hash,
)
from news.weights import apply_weighted_scores
from store import (
    StoredArticle,
    create_event,
    find_article_event_id,
    find_candidate_events,
    find_unenriched_events,
    get_event_articles,
    insert_article,
    touch_event_seen,
    update_event,
)
from util.cycle_log import CycleLog
from util.json_parse import extract_json_block
from util.llm_client import OllamaClient
from util.text import truncate_text
from util.time import utcnow

logger = logging.getLogger(__name__)

EVENT_MATCH_THRESHOLD = 0.4
MAX_ENRICHMENTS_PER_CYCLE = 200

ENRICH_SYSTEM_PROMPT = """\
You are a financial market analyst. Given multiple signals about the same market event,
produce a concise structured summary for traders. Signals come from sources with varying
reliability (SEC filings and IR press releases are most authoritative; social media
and search trends are least).

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


def _parse_enrichment(text: str) -> EnrichedEvent | None:
    raw = extract_json_block(text) or text.strip()
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
    signal: Signal,
    candidates: list,
) -> str | None:
    norm = normalize_title(signal.title)
    for event in candidates:
        if title_similarity(signal.title, event.summary) >= EVENT_MATCH_THRESHOLD:
            return event.id
        if title_similarity(signal.title, event.summary[:120]) >= EVENT_MATCH_THRESHOLD:
            return event.id
    for event in candidates:
        if normalize_title(event.summary[:80]) == norm[:80]:
            return event.id
    return None


def _signal_tickers(signal: Signal, watchlist: list[str]) -> list[str]:
    hints = list(signal.tickers_hint)
    extracted = extract_tickers(f"{signal.title} {signal.snippet}", known=watchlist)
    return list(dict.fromkeys(hints + extracted))


def _should_ingest_signal(signal: Signal, watchlist: list[str]) -> bool:
    """Drop noisy low-signal sources that lack watchlist ticker relevance."""
    watchset = {s.upper() for s in watchlist}
    tickers = _signal_tickers(signal, watchlist)

    if signal.source.startswith("polymarket"):
        if not tickers or not any(t in watchset for t in tickers):
            return False

    if signal.source.startswith("sec_edgar"):
        if not tickers or not any(t in watchset for t in tickers):
            return False
        if any(t.endswith("W") for t in tickers):
            return False

    return True


def _enrichment_priority(event_signals: list[Signal], watchlist: list[str]) -> float:
    watchset = {s.upper() for s in watchlist}
    tickers = set()
    for sig in event_signals:
        tickers.update(_signal_tickers(sig, watchlist))
    watchlist_hit = bool(tickers & watchset)
    max_weight = max((s.weight for s in event_signals), default=0.5)
    low_source = any(
        s.source.startswith(("polymarket", "reddit", "google_trends", "stocktwits"))
        for s in event_signals
    )
    score = max_weight * 10.0
    if watchlist_hit:
        score += 100.0
    if any(s.weight >= 0.9 for s in event_signals):
        score += 25.0
    if low_source:
        score -= 30.0
    return score


def _log_event_enrichment(
    log: CycleLog,
    event_id: str,
    event_signals: list[Signal],
    enriched: EnrichedEvent,
    *,
    parsed_ok: bool,
    importance: int,
    confidence: float,
) -> None:
    """Write LLM enrichment details to the news cycle log."""
    log.line(f"Event {event_id[:8]} ({len(event_signals)} article(s)):")
    for sig in event_signals:
        log.line(f"  • [{sig.source}] {truncate_text(sig.title, 120)}")
    if parsed_ok:
        log.line(
            f"  LLM → type={enriched.event_type} sentiment={enriched.sentiment} "
            f"importance={importance} confidence={confidence:.2f} "
            f"tickers={', '.join(enriched.tickers) or '—'}"
        )
        log.line(f"  Summary: {truncate_text(enriched.summary, 300)}")
    else:
        logger.warning(
            "LLM unparseable JSON for event %s; headline fallback → "
            "type=%s sentiment=%s importance=%d confidence=%.2f",
            event_id[:8],
            enriched.event_type,
            enriched.sentiment,
            importance,
            confidence,
        )


async def _enrich_event(
    llm: OllamaClient,
    signals: list[Signal],
) -> tuple[EnrichedEvent, bool]:
    lines = []
    for i, sig in enumerate(signals, 1):
        lines.append(f"Signal {i}:")
        lines.append(f"  Title: {sig.title}")
        lines.append(f"  Source: {sig.source} (weight: {sig.weight:.2f})")
        lines.append(f"  Snippet: {sig.snippet[:300]}")
        lines.append("")

    user_prompt = "Analyze these related signals:\n\n" + "\n".join(lines)

    response = await llm.chat(
        [
            {"role": "system", "content": ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = response.content or ""
    enriched = _parse_enrichment(content)
    if enriched and enriched.summary:
        return enriched, True

    title = signals[0].title if signals else "Unknown event"
    tickers = _signal_tickers(signals[0], []) if signals else []
    return (
        EnrichedEvent(
            summary=title,
            event_type="other",
            sentiment="neutral",
            importance=2,
            confidence=0.3,
            tickers=tickers,
            companies=[],
        ),
        False,
    )


def _signals_from_articles(articles: list[StoredArticle]) -> list[Signal]:
    return [
        Signal(
            url=a.url,
            title=a.title,
            source=a.source,
            published_at=a.published_at,
            snippet=a.snippet,
            weight=a.weight,
        )
        for a in articles
    ]


async def _persist_enrichment(
    llm: OllamaClient,
    db_path: str,
    event_id: str,
    event_signals: list[Signal],
    log: CycleLog,
    stats: dict[str, int],
) -> None:
    enriched, parsed_ok = await _enrich_event(llm, event_signals)
    importance, confidence = apply_weighted_scores(
        event_signals,
        importance=enriched.importance,
        confidence=enriched.confidence,
    )
    _log_event_enrichment(
        log,
        event_id,
        event_signals,
        enriched,
        parsed_ok=parsed_ok,
        importance=importance,
        confidence=confidence,
    )
    update_event(
        db_path,
        event_id,
        summary=enriched.summary,
        event_type=enriched.event_type,
        sentiment=enriched.sentiment,
        importance=importance,
        confidence=confidence,
        tickers=enriched.tickers,
        companies=enriched.companies,
        article_count=len(event_signals),
        last_seen_at=utcnow(),
    )
    stats["updated_events"] += 1


async def process_signals(
    settings: Settings,
    signals: list[Signal],
    llm: OllamaClient,
    watchlist: list[str],
    log: CycleLog,
) -> dict[str, int]:
    """Ingest new signals: dedupe, group, enrich, persist. Returns stats."""
    db_path = settings.event_store_path

    stats = {"fetched": len(signals), "new": 0, "skipped": 0, "updated_events": 0}

    if signals:
        log.section("Enrichment")

    events_to_enrich: dict[str, list[Signal]] = {}
    enrichment_logged = False

    for signal in signals:
        if not _should_ingest_signal(signal, watchlist):
            stats["skipped"] += 1
            continue

        h = url_hash(signal.url)
        existing_event_id = find_article_event_id(db_path, h)
        if existing_event_id is not None:
            touch_event_seen(db_path, existing_event_id, seen_at=utcnow())
            stats["skipped"] += 1
            continue

        tickers = _signal_tickers(signal, watchlist)
        candidates = find_candidate_events(db_path, tickers, since_hours=48)

        event_id = _match_event(signal, candidates)

        if event_id is None:
            event_id = create_event(
                db_path,
                summary=signal.title,
                tickers=tickers,
                first_seen_at=signal.published_at or utcnow(),
                last_seen_at=utcnow(),
            )

        insert_article(
            db_path,
            url=signal.url,
            url_hash=h,
            title=signal.title,
            source=signal.source,
            snippet=signal.snippet,
            event_id=event_id,
            published_at=signal.published_at,
            weight=signal.weight,
        )
        stats["new"] += 1
        log.line(
            f"New → event {event_id[:8]} [{signal.source}] "
            f"{truncate_text(signal.title, 120)}"
        )

        stored = get_event_articles(db_path, event_id)
        events_to_enrich[event_id] = _signals_from_articles(stored)

    enriched_count = 0
    enrich_queue = sorted(
        events_to_enrich.items(),
        key=lambda item: _enrichment_priority(item[1], watchlist),
        reverse=True,
    )
    for event_id, event_signals in enrich_queue:
        if enriched_count >= MAX_ENRICHMENTS_PER_CYCLE:
            if not enrichment_logged:
                log.line(
                    f"Enrichment cap reached ({MAX_ENRICHMENTS_PER_CYCLE}); "
                    "remaining events keep stub summaries"
                )
                enrichment_logged = True
            break
        await _persist_enrichment(llm, db_path, event_id, event_signals, log, stats)
        enriched_count += 1

    # Backfill stub events that already have articles but missed enrichment.
    backfill_ids = [
        eid for eid in find_unenriched_events(db_path, limit=50)
        if eid not in events_to_enrich
    ]
    for event_id in backfill_ids:
        if enriched_count >= MAX_ENRICHMENTS_PER_CYCLE:
            break
        stored = get_event_articles(db_path, event_id)
        if not stored:
            continue
        await _persist_enrichment(
            llm, db_path, event_id, _signals_from_articles(stored), log, stats
        )
        enriched_count += 1

    return stats
