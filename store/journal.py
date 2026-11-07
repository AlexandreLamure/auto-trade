"""Persist cycle decisions and later mark-to-market outcomes."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta, timezone
from hashlib import sha256
from typing import Any

from store.db import get_connection
from util.time import parse_iso, to_iso, utcnow


def brief_hash(markdown: str) -> str:
    return sha256(markdown.encode("utf-8")).hexdigest()[:16]


def record_cycle(
    db_path: str,
    *,
    cycle_id: str,
    equity: float,
    cash: float,
    month_pnl_pct: float | None,
    consensus: str,
    dissent: str | None,
    markdown: str,
    proposals: list[dict[str, Any]],
    proposed: list[dict[str, Any]],
    approved_keys: set[tuple[str, str]],
    rejections: list[str],
    fill_status: dict[tuple[str, str], str],
    prices: dict[str, float],
    fill_records: list[dict[str, Any]] | None = None,
) -> None:
    """Write one trading cycle's proposals, orders, and decision prices."""
    now = utcnow()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cycles (
                id, started_at, equity, cash, month_pnl_pct, consensus, dissent, brief_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cycle_id,
                to_iso(now),
                equity,
                cash,
                month_pnl_pct,
                consensus,
                dissent or "",
                brief_hash(markdown),
            ),
        )
        for proposal in proposals:
            conn.execute(
                """
                INSERT INTO cycle_proposals (
                    id, cycle_id, persona, round, stance, confidence, orders_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    cycle_id,
                    str(proposal.get("persona", "")),
                    2,
                    str(proposal.get("stance", "")),
                    float(proposal.get("confidence") or 0),
                    json.dumps(proposal.get("orders") or []),
                ),
            )
        executed: dict[tuple[str, str], dict[str, Any]] = {}
        for rec in fill_records or []:
            symbol = str(rec.get("symbol", "")).upper()
            side = str(rec.get("side", "")).lower()
            if symbol and side:
                executed[(symbol, side)] = rec
        for order in proposed:
            symbol = str(order.get("symbol", "")).upper()
            side = str(order.get("side", "")).lower()
            key = (symbol, side)
            rec = executed.get(key)
            if rec is not None:
                status = str(rec.get("status") or fill_status.get(key) or "submitted")
                reason = str(rec.get("reason") or "")
                qty = float(rec.get("qty") or order.get("qty") or order.get("quantity") or 0)
                limit_price = rec.get("limit_price")
            elif key in fill_status:
                status = fill_status[key]
                reason = ""
                qty = float(order.get("qty") or order.get("quantity") or 0)
                limit_price = None
            elif key in approved_keys:
                status = "approved"
                reason = ""
                qty = float(order.get("qty") or order.get("quantity") or 0)
                limit_price = None
            else:
                status = "rejected"
                reason = next((r for r in rejections if symbol in r), "")
                qty = float(order.get("qty") or order.get("quantity") or 0)
                limit_price = None
            price = prices.get(symbol)
            conn.execute(
                """
                INSERT INTO cycle_orders (
                    id, cycle_id, symbol, side, qty, rationale, status, reason,
                    decision_price, limit_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    cycle_id,
                    symbol,
                    side,
                    qty,
                    str(order.get("rationale") or ""),
                    status,
                    reason,
                    price,
                    float(limit_price) if limit_price not in (None, "") else None,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO cycle_marks (
                    cycle_id, symbol, decision_price, started_at
                ) VALUES (?, ?, ?, ?)
                """,
                (cycle_id, symbol, price, to_iso(now)),
            )
        conn.commit()


# (calendar days after the decision, price column, marked-at column)
_MARK_HORIZONS = ((1, "price_1d", "marked_1d_at"), (5, "price_5d", "marked_5d_at"),
                  (30, "price_30d", "marked_30d_at"))
# Marks older than this are final; the window covers the 30d horizon plus slack.
MARK_WINDOW_DAYS = 60


def _started_utc(value: object):
    started = parse_iso(str(value))
    if started is not None and started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started


def _open_mark_rows(conn) -> list[Any]:
    cutoff = utcnow() - timedelta(days=MARK_WINDOW_DAYS)
    rows = conn.execute("SELECT * FROM cycle_marks").fetchall()
    return [r for r in rows if (s := _started_utc(r["started_at"])) and s >= cutoff]


def pending_mark_symbols(db_path: str) -> list[str]:
    """Symbols whose decision marks are still inside the marking window."""
    with get_connection(db_path) as conn:
        return sorted({str(r["symbol"]).upper() for r in _open_mark_rows(conn)})


def update_marks(db_path: str, closes: dict[str, list[tuple[str, float]]]) -> int:
    """Set 1d/5d/30d marks to the first daily close on/after each horizon date.

    *closes* maps symbol -> [(YYYY-MM-DD, close)] in ascending date order.
    Recomputed from bars on every call, so earlier marks are corrected too.
    """
    now_iso = to_iso(utcnow())
    updated = 0
    with get_connection(db_path) as conn:
        for row in _open_mark_rows(conn):
            symbol = str(row["symbol"]).upper()
            series = closes.get(symbol)
            started = _started_utc(row["started_at"])
            if not series or started is None:
                continue
            fields: list[str] = []
            values: list[object] = []
            for days, price_col, marked_col in _MARK_HORIZONS:
                target = (started + timedelta(days=days)).date().isoformat()
                close = next((c for d, c in series if d >= target and c > 0), None)
                if close is None or row[price_col] == close:
                    continue
                fields += [f"{price_col} = ?", f"{marked_col} = ?"]
                values += [close, now_iso]
            if not fields:
                continue
            values.extend([row["cycle_id"], symbol])
            conn.execute(
                f"UPDATE cycle_marks SET {', '.join(fields)} WHERE cycle_id = ? AND symbol = ?",
                values,
            )
            updated += 1
        conn.commit()
    return updated


def _pct(price: float | None, base: float | None) -> str:
    if not price or not base:
        return "—"
    return f"{(price / base - 1.0) * 100.0:+.1f}%"


def format_decision_history(db_path: str, *, limit: int = 12) -> str:
    """Markdown table of recent committee orders and the price move afterwards."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.started_at, o.symbol, o.side, o.status, o.rationale,
                   m.decision_price, m.price_1d, m.price_5d, m.price_30d
            FROM cycle_orders o
            JOIN cycles c ON c.id = o.cycle_id
            LEFT JOIN cycle_marks m ON m.cycle_id = o.cycle_id AND m.symbol = o.symbol
            ORDER BY c.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    if not rows:
        return ""
    lines = [
        "Price change after each past committee order (executed or not). "
        "For a sell, a drop means the call was right.",
        "",
        "| Date | Symbol | Side | Status | Decision price | +1d | +5d | +30d | Rationale |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        base = r["decision_price"]
        rationale = " ".join(str(r["rationale"] or "").split())[:120]
        lines.append(
            f"| {str(r['started_at'])[:10]} | {r['symbol']} | {r['side']} | {r['status']} | "
            f"{f'${base:.2f}' if base else '—'} | {_pct(r['price_1d'], base)} | "
            f"{_pct(r['price_5d'], base)} | {_pct(r['price_30d'], base)} | {rationale} |"
        )
    return "\n".join(lines)
