"""Event store CRUD and query API."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from store.db import connect, get_connection


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


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
            first_seen_at=_parse_iso(row["first_seen_at"]) or _utcnow(),
            last_seen_at=_parse_iso(row["last_seen_at"]) or _utcnow(),
            created_at=_parse_iso(row["created_at"]) or _utcnow(),
            updated_at=_parse_iso(row["updated_at"]) or _utcnow(),
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

    @classmethod
    def from_row(cls, row: Any) -> StoredArticle:
        return cls(
            id=row["id"],
            url=row["url"],
            url_hash=row["url_hash"],
            title=row["title"],
            source=row["source"],
            published_at=_parse_iso(row["published_at"]),
            snippet=row["snippet"],
            event_id=row["event_id"],
            fetched_at=_parse_iso(row["fetched_at"]) or _utcnow(),
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
    now = _utcnow()
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
                _iso(first_seen),
                _iso(last_seen),
                _iso(now),
                _iso(now),
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
        values.append(json.dumps(tickers))
    if companies is not None:
        fields.append("companies = ?")
        values.append(json.dumps(companies))
    if article_count is not None:
        fields.append("article_count = ?")
        values.append(article_count)
    if last_seen_at is not None:
        fields.append("last_seen_at = ?")
        values.append(_iso(last_seen_at))

    if not fields:
        return

    fields.append("updated_at = ?")
    values.append(_iso(_utcnow()))
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
) -> str:
    article_id = uuid.uuid4().hex
    now = _utcnow()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO articles (
                id, url, url_hash, title, source, published_at, snippet,
                event_id, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                url,
                url_hash,
                title,
                source,
                _iso(published_at) if published_at else None,
                snippet,
                event_id,
                _iso(now),
            ),
        )
        conn.commit()
    return article_id


def get_event(db_path: str, event_id: str) -> MarketEvent | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return MarketEvent.from_row(row)


def get_event_articles(db_path: str, event_id: str) -> list[StoredArticle]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM articles WHERE event_id = ? ORDER BY published_at DESC",
            (event_id,),
        ).fetchall()
        return [StoredArticle.from_row(row) for row in rows]


def find_candidate_events(
    db_path: str,
    tickers: list[str],
    *,
    since_hours: int = 48,
) -> list[MarketEvent]:
    if not tickers:
        return find_recent_events(db_path, since_hours=since_hours, limit=50)

    cutoff = _iso(_utcnow() - timedelta(hours=since_hours))
    ticker_set = {t.upper() for t in tickers}

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM events
            WHERE last_seen_at >= ?
            ORDER BY last_seen_at DESC
            LIMIT 200
            """,
            (cutoff,),
        ).fetchall()

    candidates: list[MarketEvent] = []
    for row in rows:
        event = MarketEvent.from_row(row)
        event_tickers = {t.upper() for t in event.tickers}
        if ticker_set & event_tickers:
            candidates.append(event)
    return candidates


def find_recent_events(
    db_path: str,
    *,
    since_hours: int = 48,
    limit: int = 50,
) -> list[MarketEvent]:
    cutoff = _iso(_utcnow() - timedelta(hours=since_hours))
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM events
            WHERE last_seen_at >= ?
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    return [MarketEvent.from_row(row) for row in rows]


def query_events(
    db_path: str,
    *,
    symbols: list[str] | None = None,
    since_hours: int = 72,
    min_importance: int = 1,
    limit: int = 30,
) -> list[MarketEvent]:
    cutoff = _iso(_utcnow() - timedelta(hours=since_hours))
    symbol_set = {s.upper() for s in (symbols or [])}

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM events
            WHERE last_seen_at >= ?
              AND importance >= ?
            ORDER BY importance DESC, last_seen_at DESC
            LIMIT ?
            """,
            (cutoff, min_importance, limit * 3),
        ).fetchall()

    events = [MarketEvent.from_row(row) for row in rows]

    if not symbol_set:
        return events[:limit]

    matched: list[MarketEvent] = []
    for event in events:
        event_tickers = {t.upper() for t in event.tickers}
        if event_tickers & symbol_set:
            matched.append(event)
        if len(matched) >= limit:
            break
    return matched


def count_events(db_path: str) -> int:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"]) if row else 0
