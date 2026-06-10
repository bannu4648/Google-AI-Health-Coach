"""
Dispatch routed intents to GoogleHealthClient methods.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.database import record_health_action
from ..core.payloads import build_nutrition_data_point, normalize_router_payload
from ..core.timezone import default_query_range_utc
from ..core.types import normalize_query_payload
from ..integrations.google_health import GoogleHealthAPIError, GoogleHealthClient
from .engine import Intent

logger = logging.getLogger(__name__)

QUERY_INTENTS = {
    Intent.QUERY_HISTORY,
    Intent.QUERY_TRENDS,
    Intent.QUERY_SLEEP,
}


def default_week_range() -> tuple[str, str]:
    return default_query_range_utc(days=7)


def _extract_data_point_id(name: str) -> str | None:
    if "/dataPoints/" not in name:
        return None
    return name.rsplit("/dataPoints/", 1)[-1]


def _find_recent_nutrition_log(
    client: GoogleHealthClient,
    *,
    food_display_name: str,
) -> dict[str, Any] | None:
    start, end = default_query_range_utc(days=2)
    result = client.list_data_points(
        "nutrition-log",
        start_time=start,
        end_time=end,
        page_size=25,
    )
    needle = food_display_name.lower()
    for point in result.get("dataPoints", []):
        nutrition = point.get("nutritionLog", {})
        display = (nutrition.get("foodDisplayName") or "").lower()
        if needle in display or display in needle:
            return point
    points = result.get("dataPoints", [])
    return points[0] if points else None


def _merge_nutrition_payload(
    payload: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    nutrition = existing.get("nutritionLog", {})
    merged = dict(payload)
    if not merged.get("food_display_name"):
        merged["food_display_name"] = nutrition.get("foodDisplayName", "Logged meal")
    if merged.get("calories_kcal") is None and nutrition.get("energy"):
        merged["calories_kcal"] = nutrition["energy"].get("kcal")
    if merged.get("meal_type") is None and nutrition.get("mealType"):
        merged["meal_type"] = nutrition["mealType"]
    if merged.get("carbs_grams") is None and nutrition.get("totalCarbohydrate"):
        merged["carbs_grams"] = nutrition["totalCarbohydrate"].get("grams")
    if merged.get("fat_grams") is None and nutrition.get("totalFat"):
        merged["fat_grams"] = nutrition["totalFat"].get("grams")
    for nutrient in nutrition.get("nutrients", []):
        if nutrient.get("nutrient") == "PROTEIN" and merged.get("protein_grams") is None:
            merged["protein_grams"] = nutrient.get("quantity", {}).get("grams")
    return merged


def _create_replacement_nutrition_log(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    existing: dict[str, Any],
) -> dict[str, Any]:
    merged = _merge_nutrition_payload(payload, existing)
    data_point = build_nutrition_data_point(merged)
    result = client.create_data_point("nutrition-log", data_point)
    return {
        **result,
        "replacement_strategy": True,
        "message": (
            "Anonymous meal logs cannot be edited via the API. "
            "A corrected entry was created — please delete the older duplicate in your app."
        ),
    }


def _update_nutrition_log(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
) -> dict[str, Any]:
    food_name = payload.get("food_display_name") or "chapati"
    existing = _find_recent_nutrition_log(client, food_display_name=food_name)
    if not existing:
        return {"error": True, "message": "Could not find a recent meal log to update."}

    data_point_id = _extract_data_point_id(existing.get("name", ""))
    if not data_point_id:
        return {"error": True, "message": "Matched meal log is missing a data point id."}

    merged = _merge_nutrition_payload(payload, existing)
    patch_body = build_nutrition_data_point(merged)

    try:
        return client.patch_data_point("nutrition-log", data_point_id, patch_body)
    except GoogleHealthAPIError as exc:
        if exc.status_code in {400, 403, 500}:
            logger.warning(
                "PATCH nutrition-log failed (%s); creating replacement entry.",
                exc.status_code,
            )
            return _create_replacement_nutrition_log(
                payload, client=client, existing=existing
            )
        raise


def execute_health_action(
    intent: Intent | str,
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient | None = None,
) -> dict[str, Any] | None:
    """Dispatch router payload to the appropriate GoogleHealthClient method."""
    health_client = client or GoogleHealthClient()
    intent_value = intent if isinstance(intent, Intent) else Intent(intent)

    if intent_value in QUERY_INTENTS:
        payload = normalize_query_payload(payload, intent=intent_value.value)

    try:
        result: dict[str, Any] | None
        if intent_value == Intent.UPDATE_NUTRITION:
            result = _update_nutrition_log(payload, client=health_client)
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value in {Intent.LOG_NUTRITION, Intent.LOG_HYDRATION, Intent.LOG_WEIGHT}:
            normalized = normalize_router_payload(intent_value.value, payload)
            result = health_client.create_data_point(
                normalized["data_type"],
                normalized["data_point"],
            )
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value == Intent.QUERY_SLEEP:
            start = payload.get("start_time")
            end = payload.get("end_time")
            if not start or not end:
                start, end = default_week_range()
            result = health_client.reconcile_all_data_points(
                payload.get("data_type", "sleep"),
                start_time=start,
                end_time=end,
            )
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value == Intent.QUERY_HISTORY:
            start = payload.get("start_time")
            end = payload.get("end_time")
            if not start or not end:
                start, end = default_week_range()
            data_type = payload.get("data_type", "nutrition-log")
            result = health_client.list_all_data_points(
                data_type,
                start_time=start,
                end_time=end,
                page_size=payload.get("page_size"),
            )
            if data_type == "exercise" and not result.get("dataPoints"):
                fallback_start, fallback_end = default_week_range()
                if (fallback_start, fallback_end) != (start, end):
                    result = health_client.list_all_data_points(
                        data_type,
                        start_time=fallback_start,
                        end_time=fallback_end,
                        page_size=payload.get("page_size"),
                    )
                    result["_fallback_range"] = {
                        "reason": "initial exercise history range returned no data",
                        "start_time": fallback_start,
                        "end_time": fallback_end,
                    }
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value == Intent.QUERY_TRENDS:
            start = payload.get("start_time")
            end = payload.get("end_time")
            if not start or not end:
                start, end = default_week_range()
            data_type = payload.get("data_type", "steps")
            method = payload.get("query_method", "daily_roll_up")
            if method == "list":
                result = health_client.list_all_data_points(
                    data_type, start_time=start, end_time=end
                )
                record_health_action(intent_value.value, status="success", payload=payload, result=result)
                return result
            if method == "reconcile":
                result = health_client.reconcile_all_data_points(
                    data_type, start_time=start, end_time=end
                )
                record_health_action(intent_value.value, status="success", payload=payload, result=result)
                return result
            if method == "roll_up":
                result = health_client.roll_up(data_type, start_time=start, end_time=end)
                record_health_action(intent_value.value, status="success", payload=payload, result=result)
                return result
            result = health_client.daily_roll_up(
                data_type, start_time=start, end_time=end
            )
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        return None
    except GoogleHealthAPIError as exc:
        logger.error("Google Health API action failed: %s", exc)
        result = {"error": True, "status_code": exc.status_code, "message": exc.message}
        record_health_action(intent_value.value, status="error", payload=payload, result=result, error=exc.message)
        return result
    except Exception as exc:
        logger.exception("Unexpected health action error: %s", exc)
        result = {"error": True, "message": str(exc)}
        record_health_action(intent_value.value, status="error", payload=payload, result=result, error=str(exc))
        return result
