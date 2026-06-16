"""Persistent coaching preferences (focus, settings) in SQLite."""

from __future__ import annotations

import json
from typing import Any

from ..core.database import connect, init_db, utc_now_iso

_DEFAULT_KEY = "default"


def _get_row(conn, key: str = _DEFAULT_KEY):
    return conn.execute(
        "SELECT * FROM coaching_preferences WHERE pref_key = ?",
        (key,),
    ).fetchone()


def get_coaching_focus(*, key: str = _DEFAULT_KEY) -> str:
    init_db()
    with connect() as conn:
        row = _get_row(conn, key)
    if not row:
        return ""
    return (row["coaching_focus"] or "").strip()


def set_coaching_focus(focus: str, *, key: str = _DEFAULT_KEY) -> dict[str, Any]:
    init_db()
    now = utc_now_iso()
    cleaned = focus.strip()
    with connect() as conn:
        existing = _get_row(conn, key)
        if existing:
            conn.execute(
                """
                UPDATE coaching_preferences
                SET coaching_focus = ?, updated_at = ?
                WHERE pref_key = ?
                """,
                (cleaned, now, key),
            )
        else:
            conn.execute(
                """
                INSERT INTO coaching_preferences (pref_key, coaching_focus, settings_json, created_at, updated_at)
                VALUES (?, ?, '{}', ?, ?)
                """,
                (key, cleaned, now, now),
            )
        row = _get_row(conn, key)
    return dict(row) if row else {}


def clear_coaching_focus(*, key: str = _DEFAULT_KEY) -> None:
    set_coaching_focus("", key=key)


def format_coaching_focus_for_prompt(*, key: str = _DEFAULT_KEY) -> str:
    focus = get_coaching_focus(key=key)
    if not focus:
        return ""
    return f"COACHING FOCUS (multi-day thread): {focus}"


def detect_and_store_coaching_focus(user_text: str) -> str | None:
    """Set coaching focus when user declares a multi-day goal."""
    lowered = user_text.lower()
    triggers = (
        "help me ",
        "this week",
        "for the next",
        "working on",
        "trying to",
        "goal is to",
        "want to lose",
        "want to cut",
        "marathon",
        "meal prep",
    )
    if not any(t in lowered for t in triggers):
        return None
    if len(user_text.strip()) < 12:
        return None
    set_coaching_focus(user_text.strip()[:500])
    return user_text.strip()[:500]
