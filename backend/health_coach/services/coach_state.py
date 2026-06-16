"""Lightweight coach memory snapshot for LLM prompts."""

from __future__ import annotations

from typing import Any

from ..integrations.google_health import GoogleHealthClient
from .fitness_plans import get_relevant_active_plan, plan_adherence_summary
from .goal_progress import enrich_goals_with_progress
from .user_goals import fetch_active_goals
from .wellness_logs import fetch_recent_moods


def build_coach_state_snapshot(*, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    plan = get_relevant_active_plan()
    goals = enrich_goals_with_progress(client=client)
    moods = fetch_recent_moods(limit=1)

    plan_summary = "No active fitness plan."
    adherence = None
    if plan:
        adherence = plan_adherence_summary(plan)
        gym_count = sum(
            1
            for workout in plan.get("workouts", [])
            if "gym" in (workout.get("title") or "").lower()
            or workout.get("exercise_type") in {"STRENGTH_TRAINING", "HIIT", "CARDIO_WORKOUT"}
        )
        plan_summary = (
            f"Plan week {plan.get('week_start_hkt')}: "
            f"{adherence['completed_workouts']}/{adherence['total_workouts']} workouts done"
            + (f", ~{gym_count} gym-style sessions" if gym_count else "")
        )

    goal_lines = [
        goal.get("progress_line") or f"- {goal.get('category', 'goal')}: {goal.get('goal_text', '')}"
        for goal in goals[:3]
    ]
    goals_summary = "\n".join(goal_lines) if goal_lines else "No active goals logged."

    mood_summary = "No mood logged recently."
    if moods:
        mood_summary = f"Last mood {moods[0].get('mood_level')}/5 ({moods[0].get('logged_at_hkt', '')})"

    return {
        "plan_summary": plan_summary,
        "goals_summary": goals_summary,
        "mood_summary": mood_summary,
        "plan_adherence": adherence,
        "goals": goals,
    }


def format_coach_state_for_prompt(
    snapshot: dict[str, Any] | None = None,
    *,
    client: GoogleHealthClient | None = None,
) -> str:
    data = snapshot or build_coach_state_snapshot(client=client)
    return (
        "COACH MEMORY (local SQLite — use QUERY_FITNESS_PLAN or QUERY_COACH_DATA for details):\n"
        f"Fitness plan: {data.get('plan_summary', '')}\n"
        f"Goals: {data.get('goals_summary', '')}\n"
        f"Mood: {data.get('mood_summary', '')}"
    )
