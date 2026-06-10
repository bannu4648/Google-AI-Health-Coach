"""
Convert flat LLM router payloads into Google Health API v4 DataPoint bodies.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .timezone import format_utc_iso, now_utc, parse_to_utc, user_utc_offset_duration


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
            "mealType": payload.get("meal_type") or "UNKNOWN",
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
    return payload
