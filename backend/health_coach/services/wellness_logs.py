"""Local mood and cycle tracking until Google Health API supports Mindfulness / Women's Health."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..core.database import connect, init_db, utc_now_iso
from ..core.timezone import now_local


def log_mood(
    *,
    logged_at_hkt: str,
    mood_level: int,
    notes: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    init_db()
    row_id = str(uuid.uuid4())
    now = utc_now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO mood_logs (id, created_at, logged_at_hkt, mood_level, notes, tags_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row_id, now, logged_at_hkt, int(mood_level), notes, json.dumps(tags or [], ensure_ascii=False)),
        )
    return {
        "id": row_id,
        "logged_at_hkt": logged_at_hkt,
        "mood_level": int(mood_level),
        "notes": notes,
        "tags": tags or [],
        "sync_note": (
            "Saved locally. Google Health mood sync will be available when the API adds Mindfulness (expected Q3 2026)."
        ),
    }


def fetch_recent_moods(*, limit: int = 14) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM mood_logs ORDER BY logged_at_hkt DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json") or "[]")
        results.append(item)
    return results


def log_cycle_event(
    *,
    logged_at_hkt: str,
    event_type: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    init_db()
    row_id = str(uuid.uuid4())
    now = utc_now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO cycle_logs (id, created_at, logged_at_hkt, event_type, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (row_id, now, logged_at_hkt, event_type, json.dumps(details or {}, ensure_ascii=False)),
        )
    return {
        "id": row_id,
        "logged_at_hkt": logged_at_hkt,
        "event_type": event_type,
        "details": details or {},
        "sync_note": (
            "Saved locally. Google Health cycle sync will be available when the API adds Women's Health (expected Q3 2026)."
        ),
    }


def fetch_recent_cycle_events(*, limit: int = 30) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM cycle_logs ORDER BY logged_at_hkt DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json") or "{}")
        results.append(item)
    return results


def summarize_mood_trend(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No mood logs yet."
    avg = sum(int(item.get("mood_level", 0)) for item in entries) / len(entries)
    return f"Last {len(entries)} mood logs average {avg:.1f}/5."


def default_logged_at_hkt() -> str:
    return now_local().strftime("%Y-%m-%dT%H:%M:%S")
