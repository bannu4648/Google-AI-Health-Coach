"""Declarative intent → pipeline routing for the LangGraph coach."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..integrations.nutrition import needs_nutrition_lookup
from ..integrations.exercise import needs_exercise_calorie_lookup
from ..core.payloads import expand_nutrition_items
from .engine import Intent

PipelineName = Literal[
    "nutrition",
    "health_write",
    "health_query",
    "research",
    "local",
    "coach_only",
    "document",
    "coach_data",
    "wellness_plan",
    "day_review",
]

GraphNodeName = Literal[
    "batch_log_nutrition",
    "lookup_nutrition",
    "lookup_exercise_calories",
    "research_answer",
    "execute_health",
    "query_coach_data",
    "build_wellness_plan",
    "evaluate_day",
    "finalize_reply",
]


@dataclass(frozen=True)
class IntentCapability:
    pipeline: PipelineName
    needs_nutrition_lookup: bool = False
    needs_exercise_calorie_lookup: bool = False
    supports_batch: bool = False
    requires_confirm: bool = False
    terminal: bool = False


INTENT_CAPABILITIES: dict[str, IntentCapability] = {
    Intent.LOG_NUTRITION.value: IntentCapability(
        "nutrition", needs_nutrition_lookup=True, supports_batch=True, requires_confirm=True
    ),
    Intent.UPDATE_NUTRITION.value: IntentCapability("health_write", needs_nutrition_lookup=True),
    Intent.QUERY_NUTRITION.value: IntentCapability(
        "nutrition", needs_nutrition_lookup=True, supports_batch=True, terminal=True
    ),
    Intent.GENERAL_RESEARCH.value: IntentCapability("research", terminal=True),
    Intent.LOG_HYDRATION.value: IntentCapability("health_write"),
    Intent.LOG_WEIGHT.value: IntentCapability("health_write"),
    Intent.LOG_EXERCISE.value: IntentCapability("health_write", needs_exercise_calorie_lookup=True),
    Intent.UPDATE_EXERCISE.value: IntentCapability("health_write", needs_exercise_calorie_lookup=True),
    Intent.DELETE_NUTRITION.value: IntentCapability("health_write"),
    Intent.CREATE_FITNESS_PLAN.value: IntentCapability("local"),
    Intent.QUERY_FITNESS_PLAN.value: IntentCapability("local"),
    Intent.COMPLETE_WORKOUT.value: IntentCapability("local"),
    Intent.LOG_MOOD.value: IntentCapability("local"),
    Intent.QUERY_MOOD_HISTORY.value: IntentCapability("local"),
    Intent.LOG_CYCLE.value: IntentCapability("local"),
    Intent.QUERY_CYCLE.value: IntentCapability("local"),
    Intent.LOG_GOAL.value: IntentCapability("local"),
    Intent.UPDATE_GOAL.value: IntentCapability("local"),
    Intent.QUERY_GOALS.value: IntentCapability("local"),
    Intent.QUERY_COACH_DATA.value: IntentCapability("coach_data", terminal=True),
    Intent.BUILD_WELLNESS_PLAN.value: IntentCapability("wellness_plan", terminal=True),
    Intent.SUMMARIZE_DOCUMENT.value: IntentCapability("document", terminal=True),
    Intent.QUERY_HISTORY.value: IntentCapability("health_query"),
    Intent.QUERY_TRENDS.value: IntentCapability("health_query"),
    Intent.QUERY_SLEEP.value: IntentCapability("health_query"),
    Intent.EVALUATE_DAY.value: IntentCapability("day_review", terminal=True),
    Intent.COACHING_CHAT.value: IntentCapability("coach_only", terminal=True),
    Intent.UNDO_LAST_LOG.value: IntentCapability("local"),
}


LOCAL_COACH_INTENTS = {
    intent
    for intent, cap in INTENT_CAPABILITIES.items()
    if cap.pipeline == "local"
}


def get_capability(intent: str) -> IntentCapability:
    return INTENT_CAPABILITIES.get(intent, IntentCapability("coach_only", terminal=True))


def is_batch_nutrition(payload: dict) -> bool:
    return len(expand_nutrition_items(payload)) > 1


def route_after_intent(intent: str, payload: dict) -> GraphNodeName:
    """Single routing function replacing after_route / after_vision duplication."""
    cap = get_capability(intent)

    if cap.pipeline == "research":
        return "research_answer"
    if cap.pipeline == "coach_data":
        return "query_coach_data"
    if cap.pipeline == "wellness_plan":
        return "build_wellness_plan"
    if cap.pipeline == "day_review":
        return "evaluate_day"
    if cap.pipeline in {"document", "coach_only"}:
        return "finalize_reply"
    if cap.pipeline == "local" or cap.pipeline == "health_write":
        if is_batch_nutrition(payload) and cap.supports_batch:
            return "batch_log_nutrition"
        if needs_nutrition_lookup(intent, payload):
            return "lookup_nutrition"
        if cap.needs_exercise_calorie_lookup and needs_exercise_calorie_lookup(intent, payload):
            return "lookup_exercise_calories"
        return "execute_health"
    if cap.pipeline == "health_query":
        return "execute_health"
    if cap.pipeline == "nutrition":
        if is_batch_nutrition(payload) and cap.supports_batch:
            return "batch_log_nutrition"
        if needs_nutrition_lookup(intent, payload):
            return "lookup_nutrition"
        if intent in {Intent.LOG_NUTRITION.value, Intent.UPDATE_NUTRITION.value}:
            return "execute_health"
        return "finalize_reply"
    return "finalize_reply"


def route_after_exercise_lookup(intent: str, payload: dict) -> GraphNodeName:
    return "execute_health"


def route_after_nutrition_lookup(intent: str, payload: dict) -> GraphNodeName:
    from ..integrations.nutrition import should_skip_health_sync

    if should_skip_health_sync(intent, payload):
        return "finalize_reply"
    return "execute_health"
