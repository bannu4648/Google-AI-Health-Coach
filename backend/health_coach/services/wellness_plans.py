"""Tailored wellness plans combining Google Health history, goals, and fitness plan."""

from __future__ import annotations

from typing import Any

from ..core.database import add_coach_note
from ..core.timezone import default_query_range_utc, now_local
from ..integrations.google_health import GoogleHealthAPIError, GoogleHealthClient
from .fitness_plans import format_full_plan_for_reply, get_relevant_active_plan
from .user_goals import fetch_active_goals


def _compact_meals(items: list[dict[str, Any]], *, limit: int = 40) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items[:limit]:
        nutrition = item.get("nutritionLog", item.get("nutrition_log", {}))
        compact.append(
            {
                "name": nutrition.get("foodDisplayName") or nutrition.get("food_display_name"),
                "meal_type": nutrition.get("mealType") or nutrition.get("meal_type"),
                "calories_kcal": nutrition.get("caloriesKcal") or nutrition.get("calories_kcal"),
                "logged_at": item.get("interval", {}).get("startTime"),
            }
        )
    return compact


def _compact_exercise(items: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items[:limit]:
        exercise = item.get("exercise", {})
        compact.append(
            {
                "name": exercise.get("displayName") or exercise.get("display_name"),
                "type": exercise.get("exerciseType") or exercise.get("exercise_type"),
                "duration_minutes": exercise.get("durationMinutes") or exercise.get("duration_minutes"),
                "logged_at": item.get("interval", {}).get("startTime"),
            }
        )
    return compact


def fetch_wellness_plan_context(
    *,
    client: GoogleHealthClient | None = None,
    lookback_days: int = 21,
) -> dict[str, Any]:
    """Gather nutrition, exercise, goals, and active fitness plan for plan generation."""
    health = client or GoogleHealthClient()
    start, end = default_query_range_utc(days=lookback_days)
    context: dict[str, Any] = {
        "lookback_days": lookback_days,
        "range_utc": {"start": start, "end": end},
        "date_hkt": now_local().date().isoformat(),
    }

    try:
        meals = health.list_data_points(
            "nutrition-log",
            start_time=start,
            end_time=end,
            page_size=100,
        )
        context["nutrition"] = {
            "count": len(meals.get("dataPoints", [])),
            "meals": _compact_meals(meals.get("dataPoints", [])),
        }
    except (GoogleHealthAPIError, ValueError, KeyError):
        context["nutrition"] = {"count": 0, "meals": []}

    try:
        workouts = health.list_data_points("exercise", start_time=start, end_time=end)
        context["exercise"] = {
            "count": len(workouts.get("dataPoints", [])),
            "sessions": _compact_exercise(workouts.get("dataPoints", [])),
        }
    except (GoogleHealthAPIError, ValueError, KeyError):
        context["exercise"] = {"count": 0, "sessions": []}

    context["goals"] = fetch_active_goals(limit=5)
    fitness_plan = get_relevant_active_plan()
    if fitness_plan:
        context["fitness_plan"] = {
            "week_start_hkt": fitness_plan.get("week_start_hkt"),
            "full_plan_text": format_full_plan_for_reply(fitness_plan),
        }
    return context


def save_wellness_plan_note(*, message: str, context: dict[str, Any]) -> None:
    add_coach_note(
        "wellness_plan",
        message[:500],
        source="build_wellness_plan",
        payload={"context_summary": {
            "nutrition_count": context.get("nutrition", {}).get("count", 0),
            "exercise_count": context.get("exercise", {}).get("count", 0),
            "goals": [goal.get("goal_text") for goal in context.get("goals", [])],
        }},
    )
