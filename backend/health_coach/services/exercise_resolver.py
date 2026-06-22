"""Resolve exercise calories via Tavily + LLM, with MET fallback."""

from __future__ import annotations

from typing import Any

from ..core.payloads import enrich_exercise_log_payload, expand_exercise_items
from ..integrations.exercise import (
    needs_exercise_calorie_lookup,
    search_exercise_calories,
)


def _item_has_calories(item: dict[str, Any]) -> bool:
    try:
        return int(item.get("calories_kcal") or 0) > 0
    except (TypeError, ValueError):
        return False


def resolve_exercise_item_for_log(
    item: dict[str, Any],
    *,
    user_text: str,
    engine: Any,
    conversation_context: str = "",
    user_profile_context: str = "",
    weight_kg: float | None = None,
) -> dict[str, Any]:
    if _item_has_calories(item):
        return dict(item)
    search_result = search_exercise_calories(
        display_name=item.get("display_name") or item.get("exercise_type") or "workout",
        exercise_type=item.get("exercise_type") or "",
        duration_minutes=item.get("duration_minutes"),
        notes=item.get("notes") or "",
        weight_kg=weight_kg,
        user_message=user_text,
    )
    merged = engine.resolve_exercise_calories(
        user_text=user_text,
        payload=item,
        search_result=search_result,
        conversation_context=conversation_context,
        user_profile_context=user_profile_context,
        weight_kg=weight_kg,
    )
    return enrich_exercise_log_payload(merged, weight_kg=weight_kg)


def resolve_exercise_payload_for_log(
    payload: dict[str, Any],
    *,
    user_text: str,
    engine: Any,
    conversation_context: str = "",
    user_profile_context: str = "",
    weight_kg: float | None = None,
) -> dict[str, Any]:
    """Enrich single or batch LOG_EXERCISE payloads before Google Health write."""
    if not needs_exercise_calorie_lookup("LOG_EXERCISE", payload):
        items = expand_exercise_items(payload)
        if len(items) == 1:
            return enrich_exercise_log_payload(items[0], weight_kg=weight_kg)
        return payload

    items = expand_exercise_items(payload)
    if len(items) <= 1:
        return resolve_exercise_item_for_log(
            items[0] if items else payload,
            user_text=user_text,
            engine=engine,
            conversation_context=conversation_context,
            user_profile_context=user_profile_context,
            weight_kg=weight_kg,
        )

    resolved_items = [
        resolve_exercise_item_for_log(
            item,
            user_text=user_text,
            engine=engine,
            conversation_context=conversation_context,
            user_profile_context=user_profile_context,
            weight_kg=weight_kg,
        )
        for item in items
    ]
    merged = dict(payload)
    merged["items"] = resolved_items
    return merged
