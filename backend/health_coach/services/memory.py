"""
In-memory per-user conversation history for WhatsApp follow-ups.

Each message is handled independently by the graph unless history is injected
into the router prompt via this module.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from threading import Lock

from ..core.database import fetch_recent_messages_for_phone

MAX_TURNS = int(os.getenv("CONVERSATION_MAX_TURNS", "8"))
COACHING_MAX_TURNS = int(os.getenv("CONVERSATION_COACHING_MAX_TURNS", "16"))
COACHING_INTENTS = frozenset(
    {
        "COACHING_CHAT",
        "BUILD_WELLNESS_PLAN",
        "LOG_GOAL",
        "UPDATE_GOAL",
        "QUERY_GOALS",
        "CREATE_FITNESS_PLAN",
        "QUERY_FITNESS_PLAN",
        "LOG_NUTRITION",
        "QUERY_NUTRITION",
        "UPDATE_NUTRITION",
    }
)

_SCHEDULED_PREFIXES = (
    "Workout reminder:",
    "Mid-day check-in:",
    "Morning briefing",
    "Evening recap",
    "Week in review:",
)


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


_lock = Lock()
_histories: dict[str, deque[Turn]] = {}


def get_history(sender_phone: str) -> list[Turn]:
    with _lock:
        return list(_histories.get(sender_phone, deque()))


def _truncate_history_text(text: str, *, max_chars: int = 600) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


def format_history_for_prompt(
    sender_phone: str,
    *,
    exclude_user_text: str | None = None,
    intent: str = "",
) -> str:
    max_turns = COACHING_MAX_TURNS if intent in COACHING_INTENTS else MAX_TURNS
    persisted = fetch_recent_messages_for_phone(sender_phone, limit=max_turns * 2)
    if persisted:
        lines = ["Recent conversation (oldest first):"]
        excluded = (exclude_user_text or "").strip()
        skipped_current = False
        for row in persisted:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            if (
                not skipped_current
                and excluded
                and row["direction"] == "inbound"
                and text == excluded
            ):
                skipped_current = True
                continue
            label = _message_label(row)
            lines.append(f"{label}: {_truncate_history_text(text)}")
        if len(lines) > 1:
            return "\n".join(lines)

    turns = get_history(sender_phone)
    if not turns:
        return ""
    lines = ["Recent conversation (oldest first):"]
    for turn in turns:
        if turn.role == "coach" and any(turn.text.startswith(prefix) for prefix in _SCHEDULED_PREFIXES):
            label = "Coach (scheduled)"
        else:
            label = "User" if turn.role == "user" else "Coach"
        lines.append(f"{label}: {_truncate_history_text(turn.text)}")
    return "\n".join(lines)


def _message_label(row: dict) -> str:
    text = (row.get("text") or "").strip()
    if row["direction"] == "inbound":
        return "User"
    if any(text.startswith(prefix) for prefix in _SCHEDULED_PREFIXES):
        return "Coach (scheduled)"
    payload = row.get("payload") or {}
    if isinstance(payload, dict) and payload.get("source") == "scheduler":
        return "Coach (scheduled)"
    return "Coach"


def record_coach_outreach(sender_phone: str, *, text: str, source: str = "scheduler") -> None:
    """Record proactive coach messages so follow-ups have full thread context."""
    if not sender_phone or not text.strip():
        return
    prefix = f"[{source}] " if source != "scheduler" else ""
    append_turn(sender_phone, role="coach", text=f"{prefix}{text.strip()}")


def append_turn(sender_phone: str, *, role: str, text: str) -> None:
    if not sender_phone or not text.strip():
        return
    with _lock:
        history = _histories.setdefault(sender_phone, deque(maxlen=COACHING_MAX_TURNS * 2))
        history.append(Turn(role=role, text=text.strip()))


def record_exchange(sender_phone: str, *, user_text: str, coach_reply: str) -> None:
    append_turn(sender_phone, role="user", text=user_text)
    append_turn(sender_phone, role="coach", text=coach_reply)
