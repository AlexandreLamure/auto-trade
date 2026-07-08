"""Event store CRUD and query API."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from news.weights import UNKNOWN_SOURCE_WEIGHT
from store.db import connect, get_connection
from util.time import parse_iso, to_iso, utcnow


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
            tickers=_loads_json_list(row["tickers"]),
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
    weight: float = UNKNOWN_SOURCE_WEIGHT

    @classmethod
    def from_row(cls, row: Any) -> StoredArticle:
        keys = row.keys()
        weight = float(row["weight"]) if "weight" in keys else UNKNOWN_SOURCE_WEIGHT
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


def article_exists(db_path: str, url_hash: str) -> bool:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM articles WHERE url_hash = ? LIMIT 1",
            (url_hash,),
        ).fetchone()
        return row is not None


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
    """Bump last_seen_at when a duplicate article URL is re-ingested."""
    when = seen_at or utcnow()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE events
            SET last_seen_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (to_iso(when), to_iso(utcnow()), event_id),
        )
        conn.commit()


def _sync_event_tickers(conn: sqlite3.Connection, event_id: str, tickers: list[str]) -> None:
    conn.execute("DELETE FROM event_tickers WHERE event_id = ?", (event_id,))
    for ticker in tickers:
        sym = str(ticker).upper().strip()
        if sym:
            conn.execute(
                "INSERT OR IGNORE INTO event_tickers (event_id, ticker) VALUES (?, ?)",
                (event_id, sym),
            )


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
    tickers = tickers or []
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
        _sync_event_tickers(conn, event_id, tickers)
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
        values.append(json.dumps(tickers))
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
        if tickers is not None:
            _sync_event_tickers(conn, event_id, tickers)
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
    weight: float = UNKNOWN_SOURCE_WEIGHT,
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
        "importance DESC, last_seen_at DESC"
        if order_by_importance
        else "last_seen_at DESC"
    )
    stub_filter = "AND NOT (event_type = 'unknown' AND article_count = 0)"

    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        params: list[Any] = [cutoff, min_importance, *sorted(tickers), limit]
        sql = f"""
            SELECT DISTINCT e.*
            FROM events e
            INNER JOIN event_tickers et ON et.event_id = e.id
            WHERE e.last_seen_at >= ?
              AND e.importance >= ?
              AND et.ticker IN ({placeholders})
              {stub_filter}
            ORDER BY e.importance DESC, e.last_seen_at DESC
            LIMIT ?
        """
        with get_connection(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [MarketEvent.from_row(row) for row in rows]

    fetch_limit = limit
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM events
            WHERE last_seen_at >= ?
              AND importance >= ?
              {stub_filter}
            ORDER BY {order_clause}
            LIMIT ?
            """,
            (cutoff, min_importance, fetch_limit),
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
            SELECT et.ticker, MAX(e.importance) AS imp, MAX(e.last_seen_at) AS seen
            FROM event_tickers et
            INNER JOIN events e ON e.id = et.event_id
            WHERE e.last_seen_at >= ?
              AND e.importance >= ?
              AND NOT (e.event_type = 'unknown' AND e.article_count = 0)
            GROUP BY et.ticker
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
    """Return event ids with articles but stub metadata."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT e.id
            FROM events e
            WHERE e.event_type = 'unknown'
              AND e.article_count > 0
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
            f"DELETE FROM event_tickers WHERE event_id IN ({placeholders})",
            event_ids,
        )
        conn.execute(
            f"DELETE FROM events WHERE id IN ({placeholders})",
            event_ids,
        )
        conn.commit()
    return len(event_ids)
