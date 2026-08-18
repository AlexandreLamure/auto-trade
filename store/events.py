"""Event store CRUD and query API."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from store.db import connect, get_connection
from util.time import parse_iso, to_iso, utcnow

_DEFAULT_WEIGHT = 0.5


def _loads_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if item]


def _normalize_tickers(tickers: list[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ticker in tickers or []:
        sym = str(ticker).upper().strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


@dataclass
class MarketEvent:
    id: str
    summary: str
    event_type: str
    sentiment: str
    importance: int
    confidence: float
    tickers: list[str]
    companies: list[str]
    article_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Any) -> MarketEvent:
        return cls(
            id=row["id"],
            summary=row["summary"],
            event_type=row["event_type"],
            sentiment=row["sentiment"],
            importance=int(row["importance"]),
            confidence=float(row["confidence"]),
            tickers=_normalize_tickers(_loads_json_list(row["tickers"])),
            companies=_loads_json_list(row["companies"]),
            article_count=int(row["article_count"]),
            first_seen_at=parse_iso(row["first_seen_at"]) or utcnow(),
            last_seen_at=parse_iso(row["last_seen_at"]) or utcnow(),
            created_at=parse_iso(row["created_at"]) or utcnow(),
            updated_at=parse_iso(row["updated_at"]) or utcnow(),
        )


@dataclass
class StoredArticle:
    id: str
    url: str
    url_hash: str
    title: str
    source: str
    published_at: datetime | None
    snippet: str
    event_id: str
    fetched_at: datetime
    weight: float = _DEFAULT_WEIGHT

    @classmethod
    def from_row(cls, row: Any) -> StoredArticle:
        keys = row.keys()
        weight = float(row["weight"]) if "weight" in keys else _DEFAULT_WEIGHT
        return cls(
            id=row["id"],
            url=row["url"],
            url_hash=row["url_hash"],
            title=row["title"],
            source=row["source"],
            published_at=parse_iso(row["published_at"]),
            snippet=row["snippet"],
            event_id=row["event_id"],
            fetched_at=parse_iso(row["fetched_at"]) or utcnow(),
            weight=weight,
        )


def init_db(path: str) -> None:
    conn = connect(path)
    conn.close()


def find_article_event_id(db_path: str, url_hash: str) -> str | None:
    """Return the parent event id for an existing article URL hash."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT event_id FROM articles WHERE url_hash = ? LIMIT 1",
            (url_hash,),
        ).fetchone()
        return str(row["event_id"]) if row else None


def touch_event_seen(
    db_path: str,
    event_id: str,
    *,
    seen_at: datetime | None = None,
) -> None:
    """Bump last_seen_at forward when a duplicate article URL is re-ingested.

    last_seen_at is ingest time, never article published_at. Older timestamps
    must not rewind a live event out of the trader's lookback window.
    """
    when_iso = to_iso(seen_at or utcnow())
    now_iso = to_iso(utcnow())
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE events
            SET last_seen_at = CASE
                    WHEN last_seen_at < ? THEN ?
                    ELSE last_seen_at
                END,
                updated_at = ?
            WHERE id = ?
            """,
            (when_iso, when_iso, now_iso, event_id),
        )
        conn.commit()


def create_event(
    db_path: str,
    *,
    summary: str = "",
    event_type: str = "unknown",
    sentiment: str = "neutral",
    importance: int = 1,
    confidence: float = 0.5,
    tickers: list[str] | None = None,
    companies: list[str] | None = None,
    article_count: int = 0,
    first_seen_at: datetime | None = None,
    last_seen_at: datetime | None = None,
) -> str:
    event_id = uuid.uuid4().hex
    now = utcnow()
    first_seen = first_seen_at or now
    last_seen = last_seen_at or now
    tickers = _normalize_tickers(tickers)
    companies = companies or []

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO events (
                id, summary, event_type, sentiment, importance, confidence,
                tickers, companies, article_count, first_seen_at, last_seen_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                summary,
                event_type,
                sentiment,
                importance,
                confidence,
                json.dumps(tickers),
                json.dumps(companies),
                article_count,
                to_iso(first_seen),
                to_iso(last_seen),
                to_iso(now),
                to_iso(now),
            ),
        )
        conn.commit()
    return event_id


def update_event(
    db_path: str,
    event_id: str,
    *,
    summary: str | None = None,
    event_type: str | None = None,
    sentiment: str | None = None,
    importance: int | None = None,
    confidence: float | None = None,
    tickers: list[str] | None = None,
    companies: list[str] | None = None,
    article_count: int | None = None,
    last_seen_at: datetime | None = None,
) -> None:
    fields: list[str] = []
    values: list[Any] = []

    if summary is not None:
        fields.append("summary = ?")
        values.append(summary)
    if event_type is not None:
        fields.append("event_type = ?")
        values.append(event_type)
    if sentiment is not None:
        fields.append("sentiment = ?")
        values.append(sentiment)
    if importance is not None:
        fields.append("importance = ?")
        values.append(importance)
    if confidence is not None:
        fields.append("confidence = ?")
        values.append(confidence)
    if tickers is not None:
        fields.append("tickers = ?")
        values.append(json.dumps(_normalize_tickers(tickers)))
    if companies is not None:
        fields.append("companies = ?")
        values.append(json.dumps(companies))
    if article_count is not None:
        fields.append("article_count = ?")
        values.append(article_count)
    if last_seen_at is not None:
        fields.append("last_seen_at = ?")
        values.append(to_iso(last_seen_at))

    if not fields:
        return

    fields.append("updated_at = ?")
    values.append(to_iso(utcnow()))
    values.append(event_id)

    with get_connection(db_path) as conn:
        conn.execute(
            f"UPDATE events SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        conn.commit()


def insert_article(
    db_path: str,
    *,
    url: str,
    url_hash: str,
    title: str,
    source: str,
    snippet: str,
    event_id: str,
    published_at: datetime | None = None,
    weight: float = _DEFAULT_WEIGHT,
) -> str:
    article_id = uuid.uuid4().hex
    now = utcnow()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO articles (
                id, url, url_hash, title, source, published_at, snippet,
                event_id, fetched_at, weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                url,
                url_hash,
                title,
                source,
                to_iso(published_at) if published_at else None,
                snippet,
                event_id,
                to_iso(now),
                weight,
            ),
        )
        conn.execute(
            """
            UPDATE events
            SET article_count = article_count + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (to_iso(now), event_id),
        )
        conn.commit()
    return article_id


def get_event_articles(db_path: str, event_id: str) -> list[StoredArticle]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM articles WHERE event_id = ? ORDER BY published_at DESC",
            (event_id,),
        ).fetchall()
        return [StoredArticle.from_row(row) for row in rows]


def _query_events(
    db_path: str,
    *,
    since_hours: int,
    min_importance: int = 1,
    limit: int,
    tickers: set[str] | None = None,
    order_by_importance: bool = False,
) -> list[MarketEvent]:
    cutoff = to_iso(utcnow() - timedelta(hours=since_hours))
    order_clause = (
        "e.importance DESC, e.last_seen_at DESC"
        if order_by_importance
        else "e.last_seen_at DESC"
    )
    has_articles = "AND EXISTS (SELECT 1 FROM articles a WHERE a.event_id = e.id)"

    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        params: list[Any] = [cutoff, min_importance, *sorted(tickers), limit]
        sql = f"""
            SELECT DISTINCT e.*
            FROM events e, json_each(e.tickers) je
            WHERE e.last_seen_at >= ?
              AND e.importance >= ?
              AND UPPER(TRIM(je.value)) IN ({placeholders})
              {has_articles}
            ORDER BY e.importance DESC, e.last_seen_at DESC
            LIMIT ?
        """
        with get_connection(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [MarketEvent.from_row(row) for row in rows]

    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT e.* FROM events e
            WHERE e.last_seen_at >= ?
              AND e.importance >= ?
              {has_articles}
            ORDER BY {order_clause}
            LIMIT ?
            """,
            (cutoff, min_importance, limit),
        ).fetchall()

    return [MarketEvent.from_row(row) for row in rows]


def find_candidate_events(
    db_path: str,
    tickers: list[str],
    *,
    since_hours: int = 48,
) -> list[MarketEvent]:
    if not tickers:
        return _query_events(db_path, since_hours=since_hours, min_importance=1, limit=50)

    return _query_events(
        db_path,
        since_hours=since_hours,
        min_importance=1,
        limit=200,
        tickers={t.upper() for t in tickers},
    )


def query_events(
    db_path: str,
    *,
    symbols: list[str] | None = None,
    since_hours: int = 72,
    min_importance: int = 1,
    limit: int = 30,
) -> list[MarketEvent]:
    symbol_set = {s.upper() for s in (symbols or [])} or None
    return _query_events(
        db_path,
        since_hours=since_hours,
        min_importance=min_importance,
        limit=limit,
        tickers=symbol_set,
        order_by_importance=True,
    )


def count_events(db_path: str) -> int:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"]) if row else 0


def discover_event_tickers(
    db_path: str,
    *,
    since_hours: int = 72,
    min_importance: int = 2,
    limit: int = 20,
    exclude: set[str] | None = None,
) -> list[str]:
    """Return tickers from recent high-importance enriched events."""
    exclude = exclude or set()
    cutoff = to_iso(utcnow() - timedelta(hours=since_hours))
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT UPPER(TRIM(je.value)) AS ticker,
                   MAX(e.importance) AS imp,
                   MAX(e.last_seen_at) AS seen
            FROM events e, json_each(e.tickers) je
            WHERE e.last_seen_at >= ?
              AND e.importance >= ?
              AND EXISTS (SELECT 1 FROM articles a WHERE a.event_id = e.id)
              AND TRIM(je.value) != ''
            GROUP BY ticker
            ORDER BY imp DESC, seen DESC
            LIMIT ?
            """,
            (cutoff, min_importance, limit * 3),
        ).fetchall()

    discovered: list[str] = []
    seen = set(exclude)
    for row in rows:
        sym = str(row["ticker"]).upper()
        if sym in seen:
            continue
        seen.add(sym)
        discovered.append(sym)
        if len(discovered) >= limit:
            break
    return discovered


def find_unenriched_events(db_path: str, *, limit: int = 50) -> list[str]:
    """Return event ids that have article rows but still stub metadata."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT e.id
            FROM events e
            WHERE e.event_type = 'unknown'
              AND EXISTS (
                  SELECT 1 FROM articles a WHERE a.event_id = e.id
              )
            ORDER BY e.last_seen_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [str(row["id"]) for row in rows]


def prune_old_events(db_path: str, *, ttl_days: int = 30) -> int:
    """Delete events older than ttl_days. Returns number of events removed."""
    if ttl_days <= 0:
        return 0
    cutoff = to_iso(utcnow() - timedelta(days=ttl_days))
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM events WHERE last_seen_at < ?",
            (cutoff,),
        ).fetchall()
        event_ids = [str(row["id"]) for row in rows]
        if not event_ids:
            return 0
        placeholders = ",".join("?" for _ in event_ids)
        conn.execute(
            f"DELETE FROM articles WHERE event_id IN ({placeholders})",
            event_ids,
        )
        conn.execute(
            f"DELETE FROM events WHERE id IN ({placeholders})",
            event_ids,
        )
        conn.commit()
    return len(event_ids)
