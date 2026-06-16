"""Compute goal progress from Google Health snapshots and rollups."""

from __future__ import annotations

from typing import Any

from ..integrations.google_health import GoogleHealthClient
from .user_goals import fetch_active_goals, format_goals_for_reply


def _goal_progress_line(goal: dict[str, Any], snapshot: dict[str, Any]) -> str:
    target = goal.get("target") or {}
    category = (goal.get("category") or "").lower()
    text = goal.get("goal_text", "")

    if category == "fitness" or target.get("sessions_per_week"):
        sessions_target = int(target.get("sessions_per_week", 0) or 0)
        completed = int((goal.get("progress") or {}).get("sessions_completed", 0) or 0)
        if sessions_target:
            return f"{text}: {completed}/{sessions_target} workouts this week"
        weekly = snapshot.get("weekly_trends", snapshot)
        total = weekly.get("exercise", {}).get("total_sessions", snapshot.get("exercise", {}).get("count", 0))
        return f"{text}: {total} workouts logged this week"

    if target.get("steps_per_day") or "step" in text.lower():
        target_steps = int(target.get("steps_per_day", 10000) or 10000)
        current = int(snapshot.get("steps", {}).get("count", 0) or 0)
        gap = max(0, target_steps - current)
        if gap:
            return f"{text}: {current:,}/{target_steps:,} steps today ({gap:,} to go)"
        return f"{text}: {current:,}/{target_steps:,} steps — on track today"

    if target.get("meals_per_day") or "meal" in text.lower() or "log" in text.lower():
        target_meals = int(target.get("meals_per_day", 3) or 3)
        current = int(snapshot.get("nutrition", {}).get("count", 0) or 0)
        return f"{text}: {current}/{target_meals} meals logged today"

    progress = goal.get("progress") or {}
    if progress:
        return f"{text}: {progress}"
    return text


def enrich_goals_with_progress(
    goals: list[dict[str, Any]] | None = None,
    *,
    snapshot: dict[str, Any] | None = None,
    client: GoogleHealthClient | None = None,
) -> list[dict[str, Any]]:
    active = goals if goals is not None else fetch_active_goals(limit=5)
    if not active:
        return []
    if snapshot is None:
        from .coaching import get_daily_health_snapshot

        snap = get_daily_health_snapshot(client=client)
    else:
        snap = snapshot
    enriched: list[dict[str, Any]] = []
    for goal in active:
        item = dict(goal)
        item["progress_line"] = _goal_progress_line(goal, snap)
        enriched.append(item)
    return enriched


def format_goal_progress_for_prompt(
    goals: list[dict[str, Any]] | None = None,
    *,
    snapshot: dict[str, Any] | None = None,
    client: GoogleHealthClient | None = None,
) -> str:
    enriched = enrich_goals_with_progress(goals, snapshot=snapshot, client=client)
    if not enriched:
        return "No active goals."
    lines = ["Goal progress (today / this week):"]
    for goal in enriched:
        lines.append(f"- {goal.get('progress_line', goal.get('goal_text', ''))}")
    return "\n".join(lines)


def format_goal_progress_for_summary(
    goals: list[dict[str, Any]] | None = None,
    *,
    snapshot: dict[str, Any] | None = None,
    client: GoogleHealthClient | None = None,
) -> str:
    enriched = enrich_goals_with_progress(goals, snapshot=snapshot, client=client)
    if not enriched:
        return ""
    return " ".join(goal.get("progress_line", "") for goal in enriched if goal.get("progress_line"))
