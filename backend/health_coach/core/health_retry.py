"""
Google Health API write-error recovery: deterministic fixes + optional LLM assist.
"""

from __future__ import annotations

import re
from typing import Any

from .payloads import VALID_MEAL_TYPES, normalize_exercise_type, normalize_meal_type

_RETRYABLE_STATUS = frozenset({400})

_FIELD_ERROR_MARKERS = (
    "invalid value",
    "meal_type",
    "mealtype",
    "exercise_type",
    "exercisetype",
    "data_point",
    "enum",
    "required",
)


def is_retryable_health_api_error(status_code: int, message: str) -> bool:
    """True for client validation errors where correcting the payload may help."""
    if status_code not in _RETRYABLE_STATUS:
        return False
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _FIELD_ERROR_MARKERS)


def _extract_invalid_enum_value(error_message: str) -> str | None:
    """Parse quoted invalid enum from messages like ... \"UNKNOWN\"."""
    match = re.search(r'"([^"]+)"\s*\)?\s*$', error_message.strip())
    if match:
        return match.group(1)
    return None


def apply_deterministic_payload_fixes(
    intent: str,
    payload: dict[str, Any],
    error_message: str,
) -> dict[str, Any] | None:
    """
    Apply code-level fixes without calling the LLM.

    Returns a new payload dict if something changed, else None.
    """
    fixed = dict(payload)
    changed = False
    lowered = (error_message or "").lower()

    if intent in {"LOG_NUTRITION", "UPDATE_NUTRITION"} and (
        "meal_type" in lowered or "mealtype" in lowered
    ):
        current = fixed.get("meal_type")
        normalized = normalize_meal_type(current)
        invalid = _extract_invalid_enum_value(error_message)
        if invalid and invalid.upper() == str(current or "").upper():
            normalized = normalize_meal_type(invalid)
        if normalized != current or str(current or "").upper() not in VALID_MEAL_TYPES:
            fixed["meal_type"] = normalized
            changed = True

    if intent in {"LOG_EXERCISE", "UPDATE_EXERCISE"} and (
        "exercise_type" in lowered or "exercisetype" in lowered
    ):
        current = fixed.get("exercise_type")
        normalized = normalize_exercise_type(current)
        if normalized != current:
            fixed["exercise_type"] = normalized
            changed = True
        elif not current or current == "EXERCISE_TYPE_UNSPECIFIED":
            fixed["exercise_type"] = "CARDIO_WORKOUT"
            changed = True

    if intent in {"LOG_EXERCISE", "UPDATE_EXERCISE"} and (
        "activeenergy" in lowered
        or ("metrics_summary" in lowered and "unknown name" in lowered)
    ):
        calories = fixed.get("calories_kcal")
        if calories is None and isinstance(fixed.get("data_point"), dict):
            metrics = (fixed["data_point"].get("exercise") or {}).get("metricsSummary") or {}
            calories = (metrics.get("activeEnergy") or {}).get("kcal") or metrics.get("caloriesKcal")
        if calories:
            fixed["calories_kcal"] = calories
            changed = True
        fixed.pop("data_point", None)

    if intent == "LOG_HYDRATION" and "unit" in lowered:
        unit = str(fixed.get("unit") or "").upper()
        if unit not in {"MILLILITER", "CUP_US", "FLUID_OUNCE_US"}:
            fixed["unit"] = "MILLILITER"
            changed = True

    return fixed if changed else None
