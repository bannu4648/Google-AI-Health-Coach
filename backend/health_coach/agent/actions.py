"""
Dispatch routed intents to GoogleHealthClient methods and local wellness storage.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.database import record_health_action, record_undoable_log, fetch_latest_undoable_log, delete_undoable_log
from ..core.health_retry import (
    apply_deterministic_payload_fixes,
    is_retryable_health_api_error,
)
from ..core.payloads import build_exercise_data_point, build_nutrition_data_point, normalize_router_payload
from ..core.timezone import default_query_range_utc
from ..core.types import normalize_query_payload
from ..integrations.google_health import GoogleHealthAPIError, GoogleHealthClient
from ..services.fitness_plans import (
    complete_workout,
    format_full_plan_for_reply,
    format_workout_for_reply,
    get_relevant_active_plan,
    get_todays_workout,
    parse_day_filter,
    save_fitness_plan,
)
from ..services.user_goals import (
    fetch_active_goals,
    fetch_all_goals,
    format_goals_for_reply,
    increment_workout_goal_progress,
    log_goal,
    sync_fitness_plan_goal,
    update_goal,
)
from ..services.wellness_logs import (
    default_logged_at_hkt,
    fetch_recent_cycle_events,
    fetch_recent_moods,
    log_cycle_event,
    log_mood,
    summarize_mood_trend,
)
from .engine import Intent

logger = logging.getLogger(__name__)

QUERY_INTENTS = {
    Intent.QUERY_HISTORY,
    Intent.QUERY_TRENDS,
    Intent.QUERY_SLEEP,
}

_GOAL_META_PHRASES = (
    "help log",
    "log my goal",
    "log goals",
    "can u help",
    "can you help",
    "how do i log",
    "what goals",
)


def _is_goal_intake_message(user_text: str, payload: dict[str, Any]) -> bool:
    """True when the user is asking how to log goals, not stating a goal."""
    if payload.get("goal_text", "").strip():
        return False
    lowered = user_text.lower()
    return any(phrase in lowered for phrase in _GOAL_META_PHRASES)


def _fitness_plan_scope(payload: dict[str, Any], user_text: str) -> str:
    """today | full_week | filtered"""
    explicit = (payload.get("scope") or "").lower()
    if explicit in {"today", "full_week"}:
        return explicit
    lowered = user_text.lower()
    if any(
        phrase in lowered
        for phrase in ("today", "today's", "todays", "this morning", "tonight's workout")
    ):
        return "today"
    if any(
        phrase in lowered
        for phrase in ("the plan", "my plan", "full plan", "whole week", "entire week", "give me the plan")
    ):
        return "full_week"
    if payload.get("day_filter"):
        return "filtered"
    return "full_week"


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


def _find_recent_exercise_log(
    client: GoogleHealthClient,
    *,
    display_name: str,
) -> dict[str, Any] | None:
    start, end = default_query_range_utc(days=2)
    result = client.list_data_points(
        "exercise",
        start_time=start,
        end_time=end,
        page_size=25,
    )
    needle = display_name.lower()
    for point in result.get("dataPoints", []):
        exercise = point.get("exercise", {})
        display = (exercise.get("displayName") or "").lower()
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


def _merge_exercise_payload(payload: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    exercise = existing.get("exercise", {})
    merged = dict(payload)
    if not merged.get("display_name"):
        merged["display_name"] = exercise.get("displayName", "Workout")
    if merged.get("exercise_type") is None and exercise.get("exerciseType"):
        merged["exercise_type"] = exercise["exerciseType"]
    if merged.get("notes") is None and exercise.get("notes"):
        merged["notes"] = exercise.get("notes")
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
    user_text: str = "",
) -> dict[str, Any]:
    food_name = payload.get("food_display_name") or "chapati"
    existing = _find_recent_nutrition_log(client, food_display_name=food_name)
    if not existing:
        return {"error": True, "message": "Could not find a recent meal log to update."}

    data_point_id = _extract_data_point_id(existing.get("name", ""))
    if not data_point_id:
        return {"error": True, "message": "Matched meal log is missing a data point id."}

    merged = _merge_nutrition_payload(payload, existing)
    try:
        return _patch_data_point_with_retry(
            Intent.UPDATE_NUTRITION,
            "nutrition-log",
            data_point_id,
            merged,
            client=client,
            user_text=user_text,
        )
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


def _update_exercise_log(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
) -> dict[str, Any]:
    name = payload.get("display_name") or "workout"
    existing = _find_recent_exercise_log(client, display_name=name)
    if not existing:
        return {"error": True, "message": "Could not find a recent workout to update."}
    data_point_id = _extract_data_point_id(existing.get("name", ""))
    if not data_point_id:
        return {"error": True, "message": "Matched workout is missing a data point id."}
    merged = _merge_exercise_payload(payload, existing)
    try:
        return _patch_data_point_with_retry(
            Intent.UPDATE_EXERCISE,
            "exercise",
            data_point_id,
            merged,
            client=client,
            user_text=user_text,
        )
    except GoogleHealthAPIError as exc:
        if exc.status_code in {400, 403, 500}:
            result = client.create_data_point("exercise", build_exercise_data_point(merged))
            return {
                **result,
                "replacement_strategy": True,
                "message": "Created a corrected workout entry — delete the older duplicate in your app if needed.",
            }
        raise


def _record_document_summary(
    *,
    phone: str | None,
    filename: str,
    mime_type: str,
    summary: str,
) -> None:
    import uuid

    from ..core.database import connect, init_db, utc_now_iso

    init_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO document_summaries (id, created_at, phone, filename, mime_type, summary, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), utc_now_iso(), phone, filename, mime_type, summary, "{}"),
        )


def execute_local_coach_action(
    intent: Intent | str,
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient | None = None,
    sender_phone: str | None = None,
    user_text: str = "",
) -> dict[str, Any]:
    """Handle fitness plans, mood/cycle logs, and document-adjacent local actions."""
    intent_value = intent if isinstance(intent, Intent) else Intent(intent)
    health_client = client or GoogleHealthClient()

    if intent_value == Intent.CREATE_FITNESS_PLAN:
        from .engine import AIEngine

        engine = AIEngine()
        generated = engine.generate_fitness_plan(
            user_text=user_text or str(payload),
            payload=payload,
        )
        if generated is None:
            return {"error": True, "message": "Could not generate a fitness plan right now."}
        saved = save_fitness_plan(
            week_start_hkt=generated.week_start_hkt or payload.get("week_start_hkt"),
            goals=generated.goals or payload.get("goals", {}),
            weekly_targets=generated.weekly_targets or {},
            workouts=generated.workouts,
        )
        sync_fitness_plan_goal(
            goals=generated.goals or payload.get("goals", {}),
            week_start_hkt=saved.get("week_start_hkt", ""),
        )
        summary = generated.conversational_reply or "Your weekly fitness plan is ready."
        full_plan = format_full_plan_for_reply(saved)
        return {"plan": saved, "message": f"{summary}\n\n{full_plan}".strip()}

    if intent_value == Intent.QUERY_FITNESS_PLAN:
        day_indices = parse_day_filter(payload.get("day_filter"))
        scope = _fitness_plan_scope(payload, user_text)
        today = get_todays_workout()
        plan = get_relevant_active_plan()
        if not plan:
            return {"message": "You don't have an active fitness plan yet. Ask me to create one for the week."}
        if day_indices:
            message = format_full_plan_for_reply(plan, day_indices=day_indices)
            return {"plan": plan, "message": message}
        if scope == "today" and today:
            return {
                "today_workout": today,
                "message": format_workout_for_reply(today),
            }
        if scope == "today" and not today:
            return {
                "plan": plan,
                "message": (
                    "No workout scheduled for today. Here's your full week plan:\n\n"
                    f"{format_full_plan_for_reply(plan)}"
                ),
            }
        if today:
            message = (
                f"Today's workout:\n{format_workout_for_reply(today)}\n\n"
                f"Full week:\n{format_full_plan_for_reply(plan)}"
            )
            return {"today_workout": today, "plan": plan, "message": message}
        return {"plan": plan, "message": format_full_plan_for_reply(plan)}

    if intent_value == Intent.COMPLETE_WORKOUT:
        workout_id = payload.get("workout_id")
        workout = None
        if workout_id:
            workout = complete_workout(workout_id)
        else:
            today = get_todays_workout()
            if today:
                workout = complete_workout(today["id"])
        if not workout:
            return {"error": True, "message": "No planned workout found to mark complete."}
        increment_workout_goal_progress()
        result: dict[str, Any] = {"workout": workout, "message": f"Marked '{workout.get('title')}' complete."}
        if payload.get("log_to_google_health", True):
            exercise_payload = {
                "display_name": workout.get("title"),
                "exercise_type": workout.get("exercise_type") or "EXERCISE_CLASS",
                "duration_minutes": workout.get("duration_minutes") or 30,
            }
            sync = execute_health_action(Intent.LOG_EXERCISE, exercise_payload, client=health_client)
            result["google_health"] = sync
        return result

    if intent_value == Intent.LOG_MOOD:
        logged_at = payload.get("logged_at_hkt") or default_logged_at_hkt()
        mood_level = int(payload.get("mood_level") or 3)
        entry = log_mood(
            logged_at_hkt=logged_at,
            mood_level=mood_level,
            notes=payload.get("notes") or "",
            tags=payload.get("tags"),
        )
        return {"mood": entry, "message": f"Mood {mood_level}/5 logged. {entry['sync_note']}"}

    if intent_value == Intent.QUERY_MOOD_HISTORY:
        limit = int(payload.get("limit") or 14)
        entries = fetch_recent_moods(limit=limit)
        return {"entries": entries, "message": summarize_mood_trend(entries)}

    if intent_value == Intent.LOG_CYCLE:
        logged_at = payload.get("logged_at_hkt") or default_logged_at_hkt()
        entry = log_cycle_event(
            logged_at_hkt=logged_at,
            event_type=payload.get("event_type") or "symptom",
            details=payload.get("details"),
        )
        return {"cycle": entry, "message": f"Cycle event logged. {entry['sync_note']}"}

    if intent_value == Intent.QUERY_CYCLE:
        limit = int(payload.get("limit") or 30)
        entries = fetch_recent_cycle_events(limit=limit)
        return {
            "entries": entries,
            "message": f"Found {len(entries)} recent cycle log(s) in your local coach history.",
        }

    if intent_value == Intent.LOG_GOAL:
        if _is_goal_intake_message(user_text, payload):
            return {
                "message": (
                    "Happy to help! Tell me your specific goal — for example: "
                    "'lose weight to 68 kg by September' or 'gym twice a week'."
                )
            }
        goal_text = (payload.get("goal_text") or "").strip()
        if not goal_text:
            return {
                "message": (
                    "What goal should I save? Include the target and timeframe if you can."
                )
            }
        entry = log_goal(
            category=payload.get("category") or "habit",
            goal_text=goal_text,
            target=payload.get("target"),
            deadline_hkt=payload.get("deadline_hkt"),
            google_health_sync=payload.get("google_health_sync") or "none",
        )
        follow_up = ""
        if entry.get("category") == "weight":
            follow_up = (
                "\n\nWant a tailored plan next? Say: "
                "'build my meal and workout plan' and I'll analyze your recent logs."
            )
        return {"goal": entry, "message": f"Goal saved: {entry['goal_text']}{follow_up}"}

    if intent_value == Intent.UPDATE_GOAL:
        entry = update_goal(
            payload.get("goal_id"),
            goal_text=payload.get("goal_text"),
            target=payload.get("target"),
            progress=payload.get("progress"),
            status=payload.get("status"),
            deadline_hkt=payload.get("deadline_hkt"),
        )
        if not entry:
            return {"error": True, "message": "Could not find a goal to update."}
        return {"goal": entry, "message": f"Goal updated: {entry['goal_text']}"}

    if intent_value == Intent.QUERY_GOALS:
        status = payload.get("status")
        if status == "active":
            goals = fetch_active_goals(limit=int(payload.get("limit") or 10))
        else:
            goals = fetch_all_goals(limit=int(payload.get("limit") or 20), status=status)
        return {"goals": goals, "message": format_goals_for_reply(goals)}

    if intent_value == Intent.UNDO_LAST_LOG:
        return _undo_last_log(client=health_client)

    return {"error": True, "message": f"Unsupported local intent: {intent_value.value}"}


def _resolve_fixed_payload(
    intent_value: Intent,
    payload: dict[str, Any],
    exc: GoogleHealthAPIError,
    *,
    user_text: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """Try deterministic then LLM fixes for a failed Google Health write."""
    fixed = apply_deterministic_payload_fixes(intent_value.value, payload, exc.message)
    fix_source = "deterministic"
    if fixed is None:
        from .engine import AIEngine

        fixed = AIEngine().fix_health_payload_from_error(
            intent=intent_value.value,
            payload=payload,
            error_message=exc.message,
            user_text=user_text,
        )
        fix_source = "llm"
    if not fixed:
        return None, ""
    fix_summary = str(fixed.pop("_fix_summary", "") or "").strip()
    if fixed == payload:
        return None, fix_summary
    if not fix_summary:
        fix_summary = f"Adjusted payload after Google Health rejected {fix_source} fix."
    return fixed, fix_summary


def _create_data_point_with_retry(
    intent_value: Intent,
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
) -> dict[str, Any]:
    normalized = normalize_router_payload(intent_value.value, payload)
    try:
        result = client.create_data_point(normalized["data_type"], normalized["data_point"])
        _maybe_record_undoable(intent_value, normalized["data_type"], result)
        return result
    except GoogleHealthAPIError as exc:
        if not is_retryable_health_api_error(exc.status_code, exc.message):
            raise
        fixed_payload, fix_summary = _resolve_fixed_payload(
            intent_value, payload, exc, user_text=user_text
        )
        if fixed_payload is None:
            raise
        logger.info(
            "Retrying %s after payload fix (%s): %s",
            intent_value.value,
            fix_summary,
            exc.message,
        )
        normalized = normalize_router_payload(intent_value.value, fixed_payload)
        result = client.create_data_point(normalized["data_type"], normalized["data_point"])
        result["_health_sync_retry"] = {
            "applied": True,
            "fix_summary": fix_summary,
            "original_error": exc.message,
        }
        _maybe_record_undoable(intent_value, normalized["data_type"], result)
        return result


def _extract_resource_name(result: dict[str, Any]) -> str | None:
    name = result.get("name")
    if name:
        return str(name)
    point = result.get("dataPoint") or result.get("data_point") or {}
    return point.get("name")


def _maybe_record_undoable(intent_value: Intent, data_type: str, result: dict[str, Any]) -> None:
    if intent_value not in {
        Intent.LOG_NUTRITION,
        Intent.LOG_HYDRATION,
        Intent.LOG_WEIGHT,
        Intent.LOG_EXERCISE,
    }:
        return
    resource = _extract_resource_name(result)
    if resource:
        record_undoable_log(
            intent=intent_value.value,
            data_type=data_type,
            resource_name=resource,
            payload=result,
        )


def _undo_last_log(*, client: GoogleHealthClient) -> dict[str, Any]:
    entry = fetch_latest_undoable_log()
    if not entry:
        return {
            "message": "Nothing recent to undo. I can only undo logs created through this coach.",
        }
    resource = entry.get("resource_name")
    if not resource:
        delete_undoable_log(entry["id"])
        return {"message": "Could not find the Google Health record for that log."}
    try:
        client.delete_data_point(resource)
        delete_undoable_log(entry["id"])
        return {
            "message": f"Undone your last {entry.get('intent', 'log').replace('_', ' ').lower()}.",
        }
    except GoogleHealthAPIError as exc:
        return {
            "error": True,
            "message": f"Could not undo that log: {exc.message}",
        }


def _patch_data_point_with_retry(
    intent_value: Intent,
    data_type: str,
    data_point_id: str,
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
) -> dict[str, Any]:
    if intent_value == Intent.UPDATE_NUTRITION:
        patch_body = build_nutrition_data_point(payload)
    else:
        patch_body = build_exercise_data_point(payload)
    try:
        return client.patch_data_point(data_type, data_point_id, patch_body)
    except GoogleHealthAPIError as exc:
        if not is_retryable_health_api_error(exc.status_code, exc.message):
            raise
        fixed_payload, fix_summary = _resolve_fixed_payload(
            intent_value, payload, exc, user_text=user_text
        )
        if fixed_payload is None:
            raise
        logger.info("Retrying PATCH %s after payload fix: %s", data_type, fix_summary)
        if intent_value == Intent.UPDATE_NUTRITION:
            patch_body = build_nutrition_data_point(fixed_payload)
        else:
            patch_body = build_exercise_data_point(fixed_payload)
        result = client.patch_data_point(data_type, data_point_id, patch_body)
        result["_health_sync_retry"] = {
            "applied": True,
            "fix_summary": fix_summary,
            "original_error": exc.message,
        }
        return result


def execute_health_action(
    intent: Intent | str,
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient | None = None,
    sender_phone: str | None = None,
    user_text: str = "",
) -> dict[str, Any] | None:
    """Dispatch router payload to the appropriate GoogleHealthClient method."""
    health_client = client or GoogleHealthClient()
    intent_value = intent if isinstance(intent, Intent) else Intent(intent)

    local_intents = {
        Intent.CREATE_FITNESS_PLAN,
        Intent.QUERY_FITNESS_PLAN,
        Intent.COMPLETE_WORKOUT,
        Intent.LOG_MOOD,
        Intent.QUERY_MOOD_HISTORY,
        Intent.LOG_CYCLE,
        Intent.QUERY_CYCLE,
        Intent.LOG_GOAL,
        Intent.UPDATE_GOAL,
        Intent.QUERY_GOALS,
        Intent.UNDO_LAST_LOG,
    }
    if intent_value in local_intents:
        return execute_local_coach_action(
            intent_value,
            payload,
            client=health_client,
            sender_phone=sender_phone,
            user_text=user_text,
        )

    if intent_value in QUERY_INTENTS:
        payload = normalize_query_payload(payload, intent=intent_value.value)

    try:
        result: dict[str, Any] | None
        if intent_value == Intent.UPDATE_NUTRITION:
            result = _update_nutrition_log(payload, client=health_client, user_text=user_text)
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value == Intent.UPDATE_EXERCISE:
            result = _update_exercise_log(payload, client=health_client, user_text=user_text)
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value in {Intent.LOG_NUTRITION, Intent.LOG_HYDRATION, Intent.LOG_WEIGHT, Intent.LOG_EXERCISE}:
            result = _create_data_point_with_retry(
                intent_value,
                payload,
                client=health_client,
                user_text=user_text,
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
