"""
Convert flat LLM router payloads into Google Health API v4 DataPoint bodies.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .timezone import format_utc_iso, now_utc, parse_to_utc, user_utc_offset_duration

VALID_MEAL_TYPES = frozenset(
    {
        "MEAL_TYPE_UNSPECIFIED",
        "BEFORE_BREAKFAST",
        "BREAKFAST",
        "BEFORE_LUNCH",
        "LUNCH",
        "BEFORE_DINNER",
        "DINNER",
        "AFTER_DINNER",
        "SNACK",
        "ANYTIME",
    }
)

_MEAL_TYPE_ALIASES: dict[str, str] = {
    "UNKNOWN": "MEAL_TYPE_UNSPECIFIED",
    "UNSPECIFIED": "MEAL_TYPE_UNSPECIFIED",
    "MEAL_TYPE_UNSPECIFIED": "MEAL_TYPE_UNSPECIFIED",
    "DRINK": "SNACK",
    "DRINKS": "SNACK",
    "ALCOHOL": "SNACK",
    "BEVERAGE": "SNACK",
    "BEVERAGES": "SNACK",
    "WINE": "SNACK",
    "BEER": "SNACK",
    "COCKTAIL": "SNACK",
    "COCKTAILS": "SNACK",
}

BATCH_NUTRITION_MAX_ITEMS = 8


def normalize_meal_type(raw: str | None) -> str:
    """Map LLM meal labels to Google Health API v4 MealType enum values."""
    if not raw or not str(raw).strip():
        return "MEAL_TYPE_UNSPECIFIED"
    cleaned = str(raw).strip().upper().replace(" ", "_").replace("-", "_")
    if cleaned in _MEAL_TYPE_ALIASES:
        return _MEAL_TYPE_ALIASES[cleaned]
    lowered = str(raw).strip().lower()
    if any(word in lowered for word in ("wine", "beer", "cocktail", "drink", "whiskey", "vodka", "gin")):
        return "SNACK"
    if cleaned in VALID_MEAL_TYPES:
        return cleaned
    return "MEAL_TYPE_UNSPECIFIED"


def expand_nutrition_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand single-item or batch LOG_NUTRITION payloads into per-item dicts."""
    items = payload.get("items")
    if isinstance(items, list) and items:
        expanded: list[dict[str, Any]] = []
        for item in items[:BATCH_NUTRITION_MAX_ITEMS]:
            if not isinstance(item, dict):
                continue
            merged = dict(payload)
            merged.pop("items", None)
            merged.update(item)
            if merged.get("food_display_name"):
                expanded.append(merged)
        return expanded
    if payload.get("food_display_name"):
        return [dict(payload)]
    return []


def _utc_now() -> str:
    return format_utc_iso(now_utc())


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def resolve_session_time_utc(payload: dict[str, Any]) -> str:
    """
    Resolve a meal/drink session timestamp as UTC for the Google Health API.

    Always interpret naive timestamps as HKT. If the LLM mistakenly appends Z
    to a local HKT clock time, strip it and convert from HKT.
    """
    raw = payload.get("logged_at_hkt") or payload.get("logged_at") or payload.get("start_time")
    if not raw:
        return _utc_now()
    text = str(raw).strip().rstrip("Z")
    return format_utc_iso(parse_to_utc(text))


def _parse_utc(iso_value: str) -> datetime:
    return parse_to_utc(iso_value)


def _format_utc(dt: datetime) -> str:
    return format_utc_iso(dt)


def _session_interval(
    payload: dict[str, Any],
    *,
    default_duration_minutes: int = 15,
) -> dict[str, str]:
    """
    Build a session interval where endTime is strictly after startTime.

    Google Health rejects equal start/end timestamps on create.
    """
    start_raw = resolve_session_time_utc(payload)
    end_raw = payload.get("end_time_hkt") or payload.get("end_time")

    start_dt = _parse_utc(start_raw)
    if end_raw:
        end_text = str(end_raw).strip().rstrip("Z")
        end_dt = parse_to_utc(end_text)
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(minutes=default_duration_minutes)
    else:
        end_dt = start_dt + timedelta(minutes=default_duration_minutes)

    offset = user_utc_offset_duration()
    return {
        "startTime": _format_utc(start_dt),
        "startUtcOffset": offset,
        "endTime": _format_utc(end_dt),
        "endUtcOffset": offset,
    }


def _fix_session_interval(interval: dict[str, Any], *, duration_minutes: int = 15) -> dict[str, str]:
    """Ensure an existing interval obeys start < end."""
    start = interval.get("startTime")
    end = interval.get("endTime")
    if not start:
        return _session_interval({}, default_duration_minutes=duration_minutes)
    offset = user_utc_offset_duration()
    if not end or end <= start:
        start_dt = _parse_utc(start)
        return {
            "startTime": start,
            "startUtcOffset": offset,
            "endTime": _format_utc(start_dt + timedelta(minutes=duration_minutes)),
            "endUtcOffset": offset,
        }
    return {
        "startTime": start,
        "startUtcOffset": offset,
        "endTime": end,
        "endUtcOffset": offset,
    }


def build_nutrition_data_point(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "nutritionLog": {
            "interval": _session_interval(payload, default_duration_minutes=15),
            "foodDisplayName": payload.get("food_display_name") or "Logged meal",
            "mealType": normalize_meal_type(payload.get("meal_type")),
            "energy": {"kcal": _safe_int(payload.get("calories_kcal"))},
            "totalCarbohydrate": {"grams": _safe_float(payload.get("carbs_grams"))},
            "totalFat": {"grams": _safe_float(payload.get("fat_grams"))},
            "nutrients": [
                {
                    "nutrient": "PROTEIN",
                    "quantity": {"grams": _safe_float(payload.get("protein_grams"))},
                },
                {
                    "nutrient": "CARBOHYDRATES",
                    "quantity": {"grams": _safe_float(payload.get("carbs_grams"))},
                },
            ],
        }
    }


def build_hydration_data_point(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "hydrationLog": {
            "interval": _session_interval(payload, default_duration_minutes=1),
            "amountConsumed": {
                "milliliters": _safe_float(payload.get("milliliters"), 250),
                "userProvidedUnit": payload.get("unit") or "MILLILITER",
            },
        }
    }


def normalize_exercise_type(raw: str | None) -> str:
    """Normalize exercise type to Google Health Exercise.ExerciseType enum."""
    if not raw or not str(raw).strip():
        return "EXERCISE_TYPE_UNSPECIFIED"
    cleaned = str(raw).strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "RUN": "RUNNING",
        "RUNS": "RUNNING",
        "JOG": "RUNNING",
        "JOGGING": "RUNNING",
        "WALK": "WALKING",
        "WALKS": "WALKING",
        "GYM": "STRENGTH_TRAINING",
        "WEIGHTS": "STRENGTH_TRAINING",
        "WEIGHT_TRAINING": "STRENGTH_TRAINING",
        "LIFT": "STRENGTH_TRAINING",
        "YOGA": "YOGA",
        "PILATES": "PILATES",
        "SWIM": "SWIMMING",
        "CYCLE": "BIKING",
        "CYCLING": "BIKING",
        "HIIT": "HIIT",
        "PICKLEBALL": "PICKLEBALL",
        "TENNIS": "TENNIS",
        "CARDIO": "CARDIO_WORKOUT",
    }
    if cleaned in aliases:
        return aliases[cleaned]
    return cleaned


def _duration_seconds(payload: dict[str, Any], *, default_minutes: int = 30) -> str:
    minutes = payload.get("duration_minutes")
    if minutes is None:
        minutes = default_minutes
    seconds = max(1, int(float(minutes) * 60))
    return f"{seconds}s"


def build_exercise_data_point(payload: dict[str, Any]) -> dict[str, Any]:
    calories = _safe_int(payload.get("calories_kcal"), 0)
    metrics: dict[str, Any] = {}
    if calories > 0:
        metrics["activeEnergy"] = {"kcal": calories}
    return {
        "exercise": {
            "interval": _session_interval(payload, default_duration_minutes=int(
                payload.get("duration_minutes") or 30
            )),
            "exerciseType": normalize_exercise_type(payload.get("exercise_type")),
            "displayName": payload.get("display_name") or payload.get("food_display_name") or "Workout",
            "activeDuration": _duration_seconds(payload),
            "notes": payload.get("notes") or "",
            "metricsSummary": metrics or {"activeEnergy": {"kcal": 0}},
        }
    }


def build_weight_data_point(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "weight": {
            "sampleTime": {"physicalTime": resolve_session_time_utc(payload)},
            "weightGrams": _safe_float(payload.get("weight_grams")),
            "notes": payload.get("notes"),
        }
    }


def normalize_router_payload(intent: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    If the router already returned a full data_point, pass through.
    Otherwise build one from flat LLM fields.
    """
    if payload.get("data_point"):
        data_point = payload["data_point"]
        if "nutritionLog" in data_point:
            data_point["nutritionLog"]["interval"] = _fix_session_interval(
                data_point["nutritionLog"].get("interval", {}),
                duration_minutes=15,
            )
        if "hydrationLog" in data_point:
            data_point["hydrationLog"]["interval"] = _fix_session_interval(
                data_point["hydrationLog"].get("interval", {}),
                duration_minutes=1,
            )
        return payload

    if intent == "LOG_NUTRITION":
        return {
            "data_type": "nutrition-log",
            "data_point": build_nutrition_data_point(payload),
        }
    if intent == "LOG_HYDRATION":
        return {
            "data_type": "hydration-log",
            "data_point": build_hydration_data_point(payload),
        }
    if intent == "LOG_WEIGHT":
        return {
            "data_type": "weight",
            "data_point": build_weight_data_point(payload),
        }
    if intent == "LOG_EXERCISE":
        return {
            "data_type": "exercise",
            "data_point": build_exercise_data_point(payload),
        }
    return payload
