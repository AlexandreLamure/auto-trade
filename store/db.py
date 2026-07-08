"""SQLite connection helpers with WAL mode for concurrent read/write."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from news.weights import UNKNOWN_SOURCE_WEIGHT

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL DEFAULT 'unknown',
    sentiment TEXT NOT NULL DEFAULT 'neutral',
    importance INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 0.5,
    tickers TEXT NOT NULL DEFAULT '[]',
    companies TEXT NOT NULL DEFAULT '[]',
    article_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    url_hash TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    published_at TEXT,
    snippet TEXT NOT NULL DEFAULT '',
    event_id TEXT NOT NULL REFERENCES events(id),
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_tickers (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    PRIMARY KEY (event_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_articles_event_id ON articles(event_id);
CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url_hash);
CREATE INDEX IF NOT EXISTS idx_events_last_seen ON events(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_event_tickers_ticker ON event_tickers(ticker);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    if "weight" not in cols:
        conn.execute(
            f"ALTER TABLE articles ADD COLUMN weight REAL NOT NULL DEFAULT {UNKNOWN_SOURCE_WEIGHT}"
        )

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "event_tickers" not in tables:
        conn.executescript(
            """
            CREATE TABLE event_tickers (
                event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                ticker TEXT NOT NULL,
                PRIMARY KEY (event_id, ticker)
            );
            CREATE INDEX IF NOT EXISTS idx_event_tickers_ticker ON event_tickers(ticker);
            """
        )
    ticker_count = conn.execute("SELECT COUNT(*) FROM event_tickers").fetchone()[0]
    if ticker_count == 0:
        _backfill_event_tickers(conn)


def _backfill_event_tickers(conn: sqlite3.Connection) -> None:
    """Populate event_tickers from legacy JSON tickers column."""
    import json

    rows = conn.execute("SELECT id, tickers FROM events").fetchall()
    for row in rows:
        event_id = row[0]
        try:
            tickers = json.loads(row[1] or "[]")
        except json.JSONDecodeError:
            tickers = []
        if not isinstance(tickers, list):
            continue
        for ticker in tickers:
            sym = str(ticker).upper().strip()
            if sym:
                conn.execute(
                    "INSERT OR IGNORE INTO event_tickers (event_id, ticker) VALUES (?, ?)",
                    (event_id, sym),
                )


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
    conn.commit()


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


@contextmanager
def get_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
