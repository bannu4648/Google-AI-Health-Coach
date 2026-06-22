"""Pending user actions (e.g. nutrition lookup awaiting 'log it' confirmation)."""

from __future__ import annotations

import json
from typing import Any

from ..core.database import connect, init_db, utc_now_iso


def save_pending_nutrition(
    phone: str,
    *,
    payload: dict[str, Any],
    intent: str = "LOG_NUTRITION",
    user_text: str = "",
) -> None:
    if not phone:
        return
    init_db()
    now = utc_now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO coaching_preferences (pref_key, coaching_focus, settings_json, created_at, updated_at)
            VALUES (?, '', ?, ?, ?)
            ON CONFLICT(pref_key) DO UPDATE SET
                settings_json = excluded.settings_json,
                updated_at = excluded.updated_at
            """,
            (
                f"pending_nutrition:{phone}",
                json.dumps(
                    {
                        "type": "nutrition",
                        "intent": intent,
                        "payload": payload,
                        "user_text": user_text,
                        "saved_at": now,
                    },
                    ensure_ascii=False,
                ),
                now,
                now,
            ),
        )


def load_pending_nutrition(phone: str) -> dict[str, Any] | None:
    if not phone:
        return None
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT settings_json FROM coaching_preferences WHERE pref_key = ?",
            (f"pending_nutrition:{phone}",),
        ).fetchone()
    if not row:
        return None
    try:
        data = json.loads(row["settings_json"] or "{}")
    except (TypeError, ValueError):
        return None
    return data if data.get("type") == "nutrition" else None


def clear_pending_nutrition(phone: str) -> None:
    if not phone:
        return
    init_db()
    with connect() as conn:
        conn.execute(
            "DELETE FROM coaching_preferences WHERE pref_key = ?",
            (f"pending_nutrition:{phone}",),
        )


def is_log_followup_text(text: str) -> bool:
    lowered = text.strip().lower()
    phrases = (
        "log it",
        "log that",
        "log this",
        "yes log",
        "please log",
        "log now",
        "log it now",
        "can u log",
        "can you log",
        "save it",
        "save that",
        "add it",
        "add as a new log",
        "new log",
        "track it",
        "log the",
        "log my",
        "log pasta",
        "log that pasta",
    )
    return any(lowered == p or lowered.startswith(p + " ") or p in lowered for p in phrases)
