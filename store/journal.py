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
        for order in proposed:
            symbol = str(order.get("symbol", "")).upper()
            side = str(order.get("side", "")).lower()
            key = (symbol, side)
            if key in fill_status:
                status = fill_status[key]
                reason = ""
            elif key in approved_keys:
                status = "approved"
                reason = ""
            else:
                status = "rejected"
                reason = next((r for r in rejections if symbol in r), "")
            price = prices.get(symbol)
            qty = float(order.get("qty") or order.get("quantity") or 0)
            conn.execute(
                """
                INSERT INTO cycle_orders (
                    id, cycle_id, symbol, side, qty, rationale, status, reason, decision_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def update_marks(db_path: str, prices: dict[str, float]) -> int:
    """Fill 1d/5d/30d marks when enough time has passed and a price is available."""
    now = utcnow()
    updated = 0
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT cycle_id, symbol, started_at, decision_price,
                   price_1d, price_5d, price_30d
            FROM cycle_marks
            """
        ).fetchall()
        for row in rows:
            symbol = str(row["symbol"]).upper()
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            started = parse_iso(str(row["started_at"]))
            if started is None:
                continue
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            age = now - started
            fields: list[str] = []
            values: list[object] = []
            if row["price_1d"] is None and age >= timedelta(days=1):
                fields.append("price_1d = ?")
                values.append(price)
                fields.append("marked_1d_at = ?")
                values.append(to_iso(now))
            if row["price_5d"] is None and age >= timedelta(days=5):
                fields.append("price_5d = ?")
                values.append(price)
                fields.append("marked_5d_at = ?")
                values.append(to_iso(now))
            if row["price_30d"] is None and age >= timedelta(days=30):
                fields.append("price_30d = ?")
                values.append(price)
                fields.append("marked_30d_at = ?")
                values.append(to_iso(now))
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
