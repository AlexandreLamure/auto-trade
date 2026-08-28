"""Deduplicate signals, group into events, and LLM-enrich."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from config.settings import Settings
from news.models import Signal
from news.sources.base import (
    extract_tickers,
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
    get_article_titles_by_event,
    get_event_articles,
    insert_article,
    update_event,
)
from util.cycle_log import CycleLog
from util.json_parse import extract_json_block
from util.llm_client import OllamaClient
from util.symbols import is_tradeable_equity_symbol
from util.time import utcnow

logger = logging.getLogger(__name__)

EVENT_MATCH_THRESHOLD = 0.55
# Shared tickers make related headlines more likely to be the same story.
EVENT_MATCH_THRESHOLD_SHARED_TICKER = 0.40
MAX_ENRICHMENTS_PER_CYCLE = 200


def _ticker_sets_compatible(a: set[str], b: set[str]) -> bool:
    """Allow match when both empty (macro) or they share at least one ticker."""
    if a and b:
        return bool(a & b)
    return not a and not b


def _match_threshold(signal_tickers: set[str], event_tickers: set[str]) -> float:
    if signal_tickers & event_tickers:
        return EVENT_MATCH_THRESHOLD_SHARED_TICKER
    return EVENT_MATCH_THRESHOLD


def _best_text_similarity(title: str, texts: list[str]) -> float:
    best = 0.0
    for text in texts:
        if not text:
            continue
        score = title_similarity(title, text)
        if score > best:
            best = score
    return best


def _match_event(
    signal: Signal,
    candidates: list,
    signal_tickers: list[str],
    article_titles_by_event: dict[str, list[str]] | None = None,
) -> str | None:
    """Attach a signal to the best existing event by title + ticker overlap.

    Compares against stored article titles and the event summary (which may be
    either a stub headline or an LLM rewrite).
    """
    sig_set = {t.upper() for t in signal_tickers if t}
    titles_by_event = article_titles_by_event or {}
    best_id: str | None = None
    best_score = 0.0

    for event in candidates:
        event_tickers = {t.upper() for t in (event.tickers or []) if t}
        if not _ticker_sets_compatible(sig_set, event_tickers):
            continue

        texts = list(titles_by_event.get(event.id, []))
        if event.summary:
            texts.append(event.summary)
        score = _best_text_similarity(signal.title, texts)
        if score < _match_threshold(sig_set, event_tickers):
            continue
        if score > best_score:
            best_score = score
            best_id = event.id

    return best_id


def _cluster_pending(
    pending: list[tuple[Signal, list[str], str]],
) -> list[list[tuple[Signal, list[str], str]]]:
    """Group same-cycle unmatched signals into event clusters (union-find)."""
    n = len(pending)
    if n == 0:
        return []
    if n == 1:
        return [pending]

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    ticker_sets = [{t.upper() for t in tickers if t} for _, tickers, _ in pending]
    for i in range(n):
        for j in range(i + 1, n):
            if not _ticker_sets_compatible(ticker_sets[i], ticker_sets[j]):
                continue
            score = title_similarity(pending[i][0].title, pending[j][0].title)
            if score >= _match_threshold(ticker_sets[i], ticker_sets[j]):
                union(i, j)

    clusters: dict[int, list[tuple[Signal, list[str], str]]] = {}
    for i, item in enumerate(pending):
        clusters.setdefault(find(i), []).append(item)
    return list(clusters.values())


def _merged_tickers(cluster: list[tuple[Signal, list[str], str]]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for _, tickers, _ in cluster:
        for t in tickers:
            sym = t.upper()
            if sym not in seen:
                seen.add(sym)
                merged.append(sym)
    return merged


_JUNK_TITLE = re.compile(
    r"cd rates|high-yield savings|heloc|home equity loan|"
    r"credit card|obituar|soft retirement|visa review|"
    r"refinance rates today|best cd |mortgage and refinance|"
    r"cash back on every purchase",
    re.I,
)
_MACRO_SOURCES = (
    "rss:fed_press",
    "rss:treasury",
    "rss:investing_macro",
    "rss:fda_press",
)

ENRICH_SYSTEM_PROMPT = """\
You are a financial market analyst. Given multiple signals about the same market event,
produce a concise structured summary for traders. Signals come from sources with varying
reliability (SEC filings and IR press releases are most authoritative; social media
and search trends are least).

Assign tickers only if the company or ticker is explicitly in the signals.
Never copy a ticker from a feed name. Never use FOREX:, CRYPTO:, futures, indexes,
or made-up labels (AI-RELATED, SPACEX). Leave tickers empty for pure macro items.

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
        tickers=[str(t).upper() for t in tickers if t and is_tradeable_equity_symbol(str(t))],
        companies=[str(c) for c in companies if c],
    )


def _signal_tickers(signal: Signal, watchlist: list[str]) -> list[str]:
    hints = list(signal.tickers_hint)
    extracted = extract_tickers(f"{signal.title} {signal.snippet}", known=watchlist)
    return [
        t
        for t in dict.fromkeys(hints + extracted)
        if is_tradeable_equity_symbol(t)
    ]


def _should_ingest_signal(signal: Signal, watchlist: list[str]) -> bool:
    """Drop noisy low-signal sources that lack watchlist ticker relevance."""
    if _JUNK_TITLE.search(signal.title or ""):
        return False

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

    if not tickers:
        return signal.source.startswith(_MACRO_SOURCES)

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
        log.line(f"  • [{sig.source}] {sig.title}")
    if parsed_ok:
        log.line(
            f"  LLM → type={enriched.event_type} sentiment={enriched.sentiment} "
            f"importance={importance} confidence={confidence:.2f} "
            f"tickers={', '.join(enriched.tickers) or '—'}"
        )
        log.line(f"  Summary: {enriched.summary}")
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
        lines.append(f"  Snippet: {sig.snippet}")
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
    pending_new: list[tuple[Signal, list[str], str]] = []
    article_titles_cache: dict[str, list[str]] = {}

    for signal in signals:
        if not _should_ingest_signal(signal, watchlist):
            stats["skipped"] += 1
            continue

        h = url_hash(signal.url)
        existing_event_id = find_article_event_id(db_path, h)
        if existing_event_id is not None:
            stats["skipped"] += 1
            continue

        tickers = _signal_tickers(signal, watchlist)
        candidates = find_candidate_events(db_path, tickers, since_hours=48)
        missing_ids = [e.id for e in candidates if e.id not in article_titles_cache]
        if missing_ids:
            article_titles_cache.update(
                get_article_titles_by_event(db_path, missing_ids)
            )
        event_id = _match_event(
            signal, candidates, tickers, article_titles_cache
        )

        if event_id is None:
            pending_new.append((signal, tickers, h))
            continue

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
        article_titles_cache.setdefault(event_id, []).append(signal.title)
        log.line(
            f"Attach → event {event_id[:8]} [{signal.source}] {signal.title}"
        )
        stored = get_event_articles(db_path, event_id)
        events_to_enrich[event_id] = _signals_from_articles(stored)

    clusters = _cluster_pending(pending_new)
    clusters.sort(
        key=lambda cluster: _enrichment_priority(
            [item[0] for item in cluster], watchlist
        ),
        reverse=True,
    )
    remaining_slots = max(0, MAX_ENRICHMENTS_PER_CYCLE - len(events_to_enrich))
    for cluster in clusters:
        if remaining_slots <= 0:
            stats["skipped"] += len(cluster)
            continue

        tickers = _merged_tickers(cluster)
        first_signal = cluster[0][0]
        first_seen = first_signal.published_at or utcnow()
        for signal, _, _ in cluster[1:]:
            pub = signal.published_at
            if pub is not None and pub < first_seen:
                first_seen = pub

        event_id = create_event(
            db_path,
            summary=first_signal.title,
            tickers=tickers,
            first_seen_at=first_seen,
            last_seen_at=utcnow(),
        )
        remaining_slots -= 1
        titles: list[str] = []
        for signal, _, h in cluster:
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
            titles.append(signal.title)
            log.line(
                f"New → event {event_id[:8]} [{signal.source}] {signal.title}"
            )
        article_titles_cache[event_id] = titles
        if len(cluster) > 1:
            log.line(
                f"  clustered {len(cluster)} articles into event {event_id[:8]}"
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
            log.line(
                f"Enrichment cap reached ({MAX_ENRICHMENTS_PER_CYCLE}); "
                "remaining queued events were not created as stubs"
            )
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
