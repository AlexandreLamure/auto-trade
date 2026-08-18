"""SQLite event store shared by the news service and trading agent."""

from store.journal import record_cycle, update_marks
from store.events import (
    MarketEvent,
    StoredArticle,
    count_events,
    create_event,
    discover_event_tickers,
    find_article_event_id,
    find_candidate_events,
    find_unenriched_events,
    get_event_articles,
    init_db,
    insert_article,
    prune_old_events,
    query_events,
    touch_event_seen,
    update_event,
)

__all__ = [
    "MarketEvent",
    "StoredArticle",
    "count_events",
    "create_event",
    "discover_event_tickers",
    "find_article_event_id",
    "find_candidate_events",
    "find_unenriched_events",
    "get_event_articles",
    "init_db",
    "insert_article",
    "prune_old_events",
    "query_events",
    "record_cycle",
    "touch_event_seen",
    "update_event",
    "update_marks",
]
