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


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


_lock = Lock()
_histories: dict[str, deque[Turn]] = {}


def get_history(sender_phone: str) -> list[Turn]:
    with _lock:
        return list(_histories.get(sender_phone, deque()))


def format_history_for_prompt(sender_phone: str) -> str:
    persisted = fetch_recent_messages_for_phone(sender_phone, limit=MAX_TURNS * 2)
    if persisted:
        lines = ["Recent conversation (oldest first):"]
        for row in persisted:
            label = "User" if row["direction"] == "inbound" else "Coach"
            lines.append(f"{label}: {row['text']}")
        return "\n".join(lines)

    turns = get_history(sender_phone)
    if not turns:
        return ""
    lines = ["Recent conversation (oldest first):"]
    for turn in turns:
        label = "User" if turn.role == "user" else "Coach"
        lines.append(f"{label}: {turn.text}")
    return "\n".join(lines)


def append_turn(sender_phone: str, *, role: str, text: str) -> None:
    if not sender_phone or not text.strip():
        return
    with _lock:
        history = _histories.setdefault(sender_phone, deque(maxlen=MAX_TURNS * 2))
        history.append(Turn(role=role, text=text.strip()))


def record_exchange(sender_phone: str, *, user_text: str, coach_reply: str) -> None:
    append_turn(sender_phone, role="user", text=user_text)
    append_turn(sender_phone, role="coach", text=coach_reply)
