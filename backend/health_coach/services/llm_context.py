"""
Shared LLM prompt context: conversation history + user profile + goals.
"""

from __future__ import annotations

from typing import Any

from ..integrations.google_health import GoogleHealthClient
from .coach_state import format_coach_state_for_prompt
from .coaching_preferences import format_coaching_focus_for_prompt
from .goal_progress import format_goal_progress_for_prompt
from .memory import format_history_for_prompt
from .user_profile import fetch_user_profile_snapshot, format_user_profile_for_prompt


def build_llm_context(
    *,
    sender_phone: str = "",
    user_text: str = "",
    health_client: GoogleHealthClient | None = None,
    intent: str = "",
) -> dict[str, str]:
    """Conversation + profile + coach memory blocks for all coach LLM calls."""
    conversation_context = format_history_for_prompt(
        sender_phone,
        exclude_user_text=user_text or None,
        intent=intent,
    )
    snapshot = fetch_user_profile_snapshot(client=health_client)
    user_profile_context = format_user_profile_for_prompt(snapshot)
    coach_state_context = format_coach_state_for_prompt(client=health_client)
    goal_progress_context = format_goal_progress_for_prompt(client=health_client)
    coaching_focus_context = format_coaching_focus_for_prompt()
    merged_coach = "\n\n".join(
        part
        for part in (coach_state_context, goal_progress_context, coaching_focus_context)
        if part.strip()
    )
    return {
        "conversation_context": conversation_context,
        "user_profile_context": user_profile_context,
        "coach_state_context": merged_coach,
    }


def merge_prompt_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
