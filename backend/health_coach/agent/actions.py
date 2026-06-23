"""
Dispatch routed intents to GoogleHealthClient methods and local wellness storage.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..core.database import record_health_action, record_undoable_log, fetch_latest_undoable_log, delete_undoable_log
from ..core.health_retry import (
    apply_deterministic_payload_fixes,
    is_retryable_health_api_error,
)
from ..core.payloads import (
    build_exercise_data_point,
    build_nutrition_data_point,
    enrich_exercise_log_payload,
    expand_exercise_items,
    expand_nutrition_items,
    fix_exercise_data_point_structure,
    normalize_router_payload,
)
from ..core.timezone import default_query_range_utc, get_user_tz, parse_to_utc
from ..core.types import normalize_query_payload
from ..integrations.google_auth import GoogleAuthRequiredError
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

_BLOCKED_GENERIC_FOOD_LABELS = frozenset(
    {"pasta", "meal", "food", "logged meal", "meal from photo", "snack", "lunch", "dinner", "breakfast"}
)

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


def _nutrition_log_local_date(point: dict[str, Any]) -> str | None:
    interval = (point.get("nutritionLog") or {}).get("interval") or {}
    start_raw = interval.get("startTime")
    if not start_raw:
        return None
    return parse_to_utc(start_raw).astimezone(get_user_tz()).strftime("%Y-%m-%d")


def _nutrition_log_local_time(point: dict[str, Any]) -> str | None:
    interval = (point.get("nutritionLog") or {}).get("interval") or {}
    start_raw = interval.get("startTime")
    if not start_raw:
        return None
    return parse_to_utc(start_raw).astimezone(get_user_tz()).strftime("%H:%M")


def _pick_best_nutrition_match(
    matches: list[dict[str, Any]],
    *,
    target_logged_at_hkt: str | None = None,
) -> dict[str, Any]:
    if len(matches) == 1:
        return matches[0]
    candidates = list(matches)
    target_date = str(target_logged_at_hkt)[:10] if target_logged_at_hkt else None
    if target_date:
        wrong_day = [pt for pt in candidates if _nutrition_log_local_date(pt) != target_date]
        if wrong_day:
            candidates = wrong_day
    target_time = str(target_logged_at_hkt)[11:16] if target_logged_at_hkt and len(str(target_logged_at_hkt)) >= 16 else None
    if target_time and len(candidates) > 1:
        def _time_distance(point: dict[str, Any]) -> int:
            local_time = _nutrition_log_local_time(point) or ""
            if not local_time:
                return 9999
            th, tm = (int(target_time[:2]), int(target_time[3:5]))
            lh, lm = (int(local_time[:2]), int(local_time[3:5]))
            return abs((th * 60 + tm) - (lh * 60 + lm))

        return min(candidates, key=_time_distance)
    return candidates[0]


def _existing_logged_at_hkt(point: dict[str, Any]) -> str | None:
    interval = (point.get("nutritionLog") or {}).get("interval") or {}
    start_raw = interval.get("startTime")
    if not start_raw:
        return None
    return parse_to_utc(start_raw).astimezone(get_user_tz()).strftime("%Y-%m-%dT%H:%M:%S")


def _find_recent_nutrition_log(
    client: GoogleHealthClient,
    *,
    food_display_name: str,
    target_logged_at_hkt: str | None = None,
    search_days: int = 7,
    exclude_names: set[str] | None = None,
) -> dict[str, Any] | None:
    start, end = default_query_range_utc(days=search_days)
    result = client.list_all_data_points(
        "nutrition-log",
        start_time=start,
        end_time=end,
        page_size=100,
    )
    needle = food_display_name.lower()
    excluded = exclude_names or set()
    matches: list[dict[str, Any]] = []
    for point in result.get("dataPoints", []):
        if point.get("name") in excluded:
            continue
        nutrition = point.get("nutritionLog", {})
        display = (nutrition.get("foodDisplayName") or "").lower()
        if needle in display or display in needle or _foods_match_for_dedupe(
            food_display_name, display
        ):
            matches.append(point)
    if not matches:
        return None
    return _pick_best_nutrition_match(matches, target_logged_at_hkt=target_logged_at_hkt)


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
    if not merged.get("logged_at_hkt"):
        existing_hkt = _existing_logged_at_hkt(existing)
        if existing_hkt:
            merged["logged_at_hkt"] = existing_hkt
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
    from ..services.coaching import clear_health_snapshot_cache

    merged = _merge_nutrition_payload(payload, existing)
    data_point = build_nutrition_data_point(merged)
    result = client.create_data_point("nutrition-log", data_point)
    food_label = merged.get("food_display_name") or "meal"
    old_name = existing.get("name")
    deleted_old = False
    if old_name:
        try:
            client.batch_delete_data_points([old_name])
            clear_health_snapshot_cache()
            deleted_old = True
        except GoogleHealthAPIError as exc:
            logger.warning("Could not delete replaced nutrition log %s: %s", old_name, exc.message)

    if deleted_old:
        message = f"Moved {food_label} to the corrected date/time in Google Health."
    else:
        message = (
            f"Google Health would not edit {food_label} in place, so I added a corrected copy. "
            "Please delete the older duplicate in your app if both still show."
        )
    return {
        **result,
        "replacement_strategy": True,
        "deleted_old_entry": deleted_old,
        "message": message,
    }


def _update_nutrition_log(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
    exclude_names: set[str] | None = None,
) -> dict[str, Any]:
    food_name = payload.get("food_display_name")
    if not food_name:
        return {"error": True, "message": "Which meal should I update? Specify food_display_name."}
    if not payload.get("logged_at_hkt"):
        return {
            "error": True,
            "message": f"Missing corrected logged_at_hkt for {food_name!r}.",
        }
    existing = _find_recent_nutrition_log(
        client,
        food_display_name=food_name,
        target_logged_at_hkt=payload.get("logged_at_hkt"),
        exclude_names=exclude_names,
    )
    if not existing:
        return {
            "error": True,
            "message": f"Could not find a recent log matching {food_name!r} to update.",
        }

    existing_name = existing.get("name") or ""
    data_point_id = _extract_data_point_id(existing_name)
    if not data_point_id:
        return {"error": True, "message": "Matched meal log is missing a data point id."}

    merged = _merge_nutrition_payload(payload, existing)
    try:
        result = _patch_data_point_with_retry(
            Intent.UPDATE_NUTRITION,
            "nutrition-log",
            data_point_id,
            merged,
            client=client,
            user_text=user_text,
        )
        touched = {existing_name}
        created = (result.get("response") or result).get("name") if isinstance(result, dict) else None
        if created:
            touched.add(created)
        result["_touched_resource_names"] = list(touched)
        return result
    except GoogleHealthAPIError as exc:
        if exc.status_code in {400, 403, 500}:
            logger.warning(
                "PATCH nutrition-log failed (%s); creating replacement entry.",
                exc.status_code,
            )
            result = _create_replacement_nutrition_log(
                payload, client=client, existing=existing
            )
            touched = {existing_name}
            created = (result.get("response") or result).get("name") if isinstance(result, dict) else None
            if created:
                touched.add(created)
            result["_touched_resource_names"] = list(touched)
            return result
        raise


def _update_nutrition_logs(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
) -> dict[str, Any]:
    items = expand_nutrition_items(payload)
    if not items:
        return {"error": True, "message": "No meals specified to update."}
    if len(items) == 1:
        return _update_nutrition_log(items[0], client=client, user_text=user_text)

    updated = 0
    replaced = 0
    errors: list[str] = []
    exclude: set[str] = set()
    for item in items:
        result = _update_nutrition_log(
            item,
            client=client,
            user_text=user_text,
            exclude_names=exclude,
        )
        for name in result.get("_touched_resource_names") or []:
            if name:
                exclude.add(name)
        label = item.get("food_display_name") or "meal"
        if result.get("error"):
            errors.append(f"{label}: {result.get('message', 'update failed')}")
        elif result.get("replacement_strategy"):
            replaced += 1
        else:
            updated += 1

    if not updated and not replaced:
        return {"error": True, "message": "; ".join(errors) or "Could not update any meals."}

    parts: list[str] = []
    if updated:
        parts.append(f"Updated {updated} meal(s) in Google Health.")
    if replaced:
        parts.append(f"Re-created {replaced} meal(s) on the corrected date (removed old entries when possible).")
    if errors:
        parts.append("Could not update: " + "; ".join(errors))
    return {
        "message": " ".join(parts),
        "partial_error": bool(errors),
        "updated_count": updated,
        "replaced_count": replaced,
    }


def _delete_nutrition_logs(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
) -> dict[str, Any]:
    """Remove duplicate meal logs; keep one entry matching keep_display_name."""
    from ..services.coaching import clear_health_snapshot_cache, local_day_bounds_utc
    from ..core.timezone import get_user_tz, parse_to_utc
    from datetime import datetime

    keywords = payload.get("match_keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    if not keywords and payload.get("food_display_name"):
        keywords = [payload["food_display_name"]]
    if not keywords:
        return {
            "error": True,
            "message": "Which meal should I delete? Tell me the food name or keywords to match.",
        }

    keep_needle = (payload.get("keep_display_name") or "").lower().strip()
    delete_all = bool(payload.get("delete_all_matches") or payload.get("delete_all"))

    date_hkt = payload.get("date_hkt")
    if date_hkt:
        day = datetime.strptime(str(date_hkt)[:10], "%Y-%m-%d").replace(tzinfo=get_user_tz())
        start, end = local_day_bounds_utc(day)
    else:
        start, end = default_query_range_utc(days=3)

    result = client.list_all_data_points(
        "nutrition-log",
        start_time=start,
        end_time=end,
        page_size=100,
    )
    points = result.get("dataPoints", [])
    keywords_lower = [str(k).lower() for k in keywords if k]

    def _matches(point: dict[str, Any]) -> bool:
        name = (point.get("nutritionLog", {}).get("foodDisplayName") or "").lower()
        return any(k in name for k in keywords_lower)

    matches = [pt for pt in points if _matches(pt)]
    if not matches:
        return {"error": True, "message": f"No meals matching {keywords_lower!r} in that range."}

    if delete_all and not keep_needle:
        names = [pt["name"] for pt in matches if pt.get("name")]
        if not names:
            return {"error": True, "message": "No deletable meal entries found."}
        client.batch_delete_data_points(names)
        clear_health_snapshot_cache()
        deleted_names = [
            pt.get("nutritionLog", {}).get("foodDisplayName", "?") for pt in matches
        ]
        return {
            "message": f"Deleted {len(names)} meal(s): {', '.join(deleted_names)}.",
            "deleted_count": len(names),
        }

    if not keep_needle:
        return {"error": True, "message": "Which meal should I keep? Specify keep_display_name."}

    keepers = [
        pt
        for pt in matches
        if keep_needle in (pt.get("nutritionLog", {}).get("foodDisplayName") or "").lower()
    ]
    if not keepers:
        return {
            "error": True,
            "message": f"Found {len(matches)} match(es) but none named like {keep_needle!r}.",
        }

    keep_time = payload.get("keep_logged_at_hkt")
    keeper = keepers[0]
    if keep_time and len(keepers) > 1:
        target = str(keep_time).strip()[:16]
        for pt in keepers:
            interval = (pt.get("nutritionLog") or {}).get("interval") or {}
            start_raw = interval.get("startTime")
            if not start_raw:
                continue
            local = parse_to_utc(start_raw).astimezone(get_user_tz()).strftime("%Y-%m-%dT%H:%M")
            if local.endswith(target[-5:]) or target in local:
                keeper = pt
                break

    to_delete = [pt for pt in matches if pt.get("name") != keeper.get("name")]
    if not to_delete:
        kept_name = keeper.get("nutritionLog", {}).get("foodDisplayName", "meal")
        return {"message": f"Only one matching entry for {kept_name!r} — nothing to delete."}

    names = [pt["name"] for pt in to_delete if pt.get("name")]
    client.batch_delete_data_points(names)
    clear_health_snapshot_cache()
    kept_name = keeper.get("nutritionLog", {}).get("foodDisplayName", "meal")
    deleted_names = [
        pt.get("nutritionLog", {}).get("foodDisplayName", "?") for pt in to_delete
    ]
    return {
        "message": (
            f"Deleted {len(names)} duplicate meal(s) ({', '.join(deleted_names)}). "
            f"Kept: {kept_name}."
        ),
        "deleted_count": len(names),
        "kept": kept_name,
    }


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
            result = client.create_data_point(
                "exercise",
                build_exercise_data_point(merged, weight_kg=_user_weight_kg(client)),
            )
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
            category=payload.get("category"),
            target=payload.get("target"),
            progress=payload.get("progress"),
            status=payload.get("status"),
            deadline_hkt=payload.get("deadline_hkt"),
        )
        if not entry:
            return {"error": True, "message": "Could not find a goal to update."}
        if entry.get("category") == "nutrition":
            goal_target = entry.get("target") or {}
            if goal_target.get("protein_grams_min") or goal_target.get("daily_calories_target"):
                from ..services.nutrition_plan import build_nutrition_plan, save_nutrition_plan_settings

                plan = build_nutrition_plan(goals=fetch_active_goals(limit=10))
                if goal_target.get("protein_grams_min"):
                    plan["protein_grams_min"] = int(goal_target["protein_grams_min"])
                if goal_target.get("protein_grams_max"):
                    plan["protein_grams_max"] = int(goal_target["protein_grams_max"])
                if goal_target.get("daily_calories_target"):
                    plan["daily_calories_target"] = int(goal_target["daily_calories_target"])
                save_nutrition_plan_settings(plan)
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


def _user_weight_kg(client: GoogleHealthClient) -> float:
    from ..services.user_profile import fetch_user_profile_snapshot

    snapshot = fetch_user_profile_snapshot(client=client)
    weight = snapshot.get("weight_kg")
    return float(weight) if weight else 70.0


def _exercise_items_have_calories(payload: dict[str, Any]) -> bool:
    items = expand_exercise_items(payload)
    if not items:
        return False
    for item in items:
        try:
            if int(item.get("calories_kcal") or 0) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _prepare_exercise_payload_if_needed(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
) -> dict[str, Any]:
    """Tavily + LLM calorie resolve for paths that bypass the graph exercise lookup node."""
    weight = _user_weight_kg(client)
    if _exercise_items_have_calories(payload):
        items = expand_exercise_items(payload)
        if len(items) <= 1:
            return enrich_exercise_log_payload(items[0] if items else payload, weight_kg=weight)
        return payload
    from ..services.exercise_resolver import resolve_exercise_payload_for_log
    from ..services.user_profile import fetch_user_profile_snapshot, format_user_profile_for_prompt
    from .engine import AIEngine

    profile = format_user_profile_for_prompt(fetch_user_profile_snapshot(client=client))
    return resolve_exercise_payload_for_log(
        payload,
        user_text=user_text,
        engine=AIEngine(),
        user_profile_context=profile,
        weight_kg=weight,
    )


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


def _normalize_food_label(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


_FOOD_DEDUPE_STOPWORDS = frozenset(
    {
        "with",
        "and",
        "the",
        "one",
        "two",
        "for",
        "from",
        "your",
        "mcdonald",
        "mcdonalds",
        "hot",
    }
)


def _food_match_tokens(name: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_food_label(name))
        if len(token) > 2 and token not in _FOOD_DEDUPE_STOPWORDS
    }


def _foods_match_for_dedupe(left: str, right: str) -> bool:
    left_norm = _normalize_food_label(left)
    right_norm = _normalize_food_label(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return True
    left_tokens = _food_match_tokens(left_norm)
    right_tokens = _food_match_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap >= min(len(left_tokens), len(right_tokens), 2)


def _is_blocked_generic_nutrition_label(name: str) -> bool:
    cleaned = _normalize_food_label(name)
    return not cleaned or cleaned in _BLOCKED_GENERIC_FOOD_LABELS


def _find_same_day_nutrition_duplicate(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
) -> dict[str, Any] | None:
    """Skip creating another log when the same meal is already on the books today."""
    from datetime import datetime

    from ..services.coaching import local_day_bounds_utc

    food = (payload.get("food_display_name") or "").strip()
    if not food:
        return None
    try:
        calories = int(payload.get("calories_kcal") or 0)
    except (TypeError, ValueError):
        calories = 0

    logged_at = payload.get("logged_at_hkt")
    if logged_at:
        day = parse_to_utc(str(logged_at).strip()).astimezone(get_user_tz())
    else:
        day = datetime.now(tz=get_user_tz())
    start, end = local_day_bounds_utc(day)
    result = client.list_all_data_points(
        "nutrition-log",
        start_time=start,
        end_time=end,
        page_size=100,
    )
    for point in result.get("dataPoints", []):
        nutrition = point.get("nutritionLog") or {}
        existing_name = _normalize_food_label(nutrition.get("foodDisplayName") or "")
        if not _foods_match_for_dedupe(food, existing_name):
            continue
        existing_kcal = int((nutrition.get("energy") or {}).get("kcal") or 0)
        if calories > 0 and existing_kcal > 0:
            tolerance = max(50, int(calories * 0.1))
            if abs(existing_kcal - calories) > tolerance:
                continue
        return point
    return None


def _create_data_point_with_retry(
    intent_value: Intent,
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
    max_attempts: int = 3,
) -> dict[str, Any]:
    if intent_value == Intent.LOG_NUTRITION:
        food_name = (payload.get("food_display_name") or "").strip()
        if _is_blocked_generic_nutrition_label(food_name):
            raise GoogleHealthAPIError(
                400,
                (
                    f"I need a more specific meal name than '{food_name or 'meal'}' before I can log it. "
                    "Describe the dish or share a photo."
                ),
            )
        duplicate = _find_same_day_nutrition_duplicate(payload, client=client)
        if duplicate:
            nutrition = duplicate.get("nutritionLog") or {}
            label = nutrition.get("foodDisplayName") or food_name or "meal"
            kcal = (nutrition.get("energy") or {}).get("kcal")
            protein = next(
                (
                    nutrient.get("quantity", {}).get("grams")
                    for nutrient in nutrition.get("nutrients") or []
                    if nutrient.get("nutrient") == "PROTEIN"
                ),
                None,
            )
            parts = [f"Already logged today: {label}"]
            if kcal:
                parts.append(f"~{kcal} kcal")
            if protein:
                parts.append(f"{int(round(float(protein)))}g protein")
            return {
                "name": duplicate.get("name"),
                "nutritionLog": nutrition,
                "_duplicate_skipped": True,
                "message": f"{' | '.join(parts)}. No duplicate added.",
            }

    current_payload = dict(payload)
    last_exc: GoogleHealthAPIError | None = None

    for attempt in range(max_attempts):
        normalized = normalize_router_payload(intent_value.value, current_payload)
        data_type = normalized["data_type"]
        data_point = normalized["data_point"]
        try:
            result = client.create_data_point(data_type, data_point)
            _maybe_record_undoable(intent_value, data_type, result)
            if attempt > 0 and last_exc is not None:
                result["_health_sync_retry"] = {
                    "applied": True,
                    "fix_summary": "Adjusted workout format and synced successfully.",
                    "original_error": last_exc.message,
                    "attempts": attempt + 1,
                }
            return result
        except GoogleHealthAPIError as exc:
            last_exc = exc
            if not is_retryable_health_api_error(exc.status_code, exc.message):
                raise
            if attempt >= max_attempts - 1:
                raise

            fixed_point = fix_exercise_data_point_structure(data_point, exc.message)
            if fixed_point and intent_value == Intent.LOG_EXERCISE:
                try:
                    result = client.create_data_point(data_type, fixed_point)
                    _maybe_record_undoable(intent_value, data_type, result)
                    result["_health_sync_retry"] = {
                        "applied": True,
                        "fix_summary": "Fixed exercise metrics format and synced successfully.",
                        "original_error": exc.message,
                        "attempts": attempt + 1,
                    }
                    return result
                except GoogleHealthAPIError as structural_exc:
                    last_exc = structural_exc
                    exc = structural_exc

            fixed_payload, fix_summary = _resolve_fixed_payload(
                intent_value, current_payload, exc, user_text=user_text
            )
            if fixed_payload is None:
                raise
            logger.info(
                "Retrying %s after payload fix (%s): %s",
                intent_value.value,
                fix_summary,
                exc.message,
            )
            current_payload = fixed_payload

    if last_exc is not None:
        raise last_exc
    raise GoogleHealthAPIError(400, "Failed to create data point after retries")


def _create_exercise_logs_with_retry(
    payload: dict[str, Any],
    *,
    client: GoogleHealthClient,
    user_text: str = "",
) -> dict[str, Any]:
    payload = _prepare_exercise_payload_if_needed(payload, client=client, user_text=user_text)
    items = expand_exercise_items(payload)
    if len(items) <= 1:
        return _create_data_point_with_retry(
            Intent.LOG_EXERCISE,
            items[0] if items else payload,
            client=client,
            user_text=user_text,
        )

    logged: list[str] = []
    errors: list[str] = []
    last_result: dict[str, Any] | None = None
    for item in items:
        name = item.get("display_name") or "workout"
        try:
            last_result = _create_data_point_with_retry(
                Intent.LOG_EXERCISE,
                item,
                client=client,
                user_text=user_text,
            )
            logged.append(str(name))
        except GoogleHealthAPIError as exc:
            errors.append(f"{name}: {exc.message}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if errors and not logged:
        return {
            "error": True,
            "message": "; ".join(errors),
        }
    if errors:
        return {
            "message": f"Logged {len(logged)} workout(s): {', '.join(logged)}.",
            "partial_error": True,
            "errors": errors,
            "logged_count": len(logged),
        }
    return {
        "message": f"Logged {len(logged)} workout(s) to Google Health: {', '.join(logged)}.",
        "logged_count": len(logged),
        **(last_result or {}),
    }


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
        weight_kg = _user_weight_kg(client)
        patch_body = build_exercise_data_point(payload, weight_kg=weight_kg)
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
            patch_body = build_exercise_data_point(fixed_payload, weight_kg=_user_weight_kg(client))
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
            result = _update_nutrition_logs(payload, client=health_client, user_text=user_text)
            status = "error" if result.get("error") else "success"
            record_health_action(
                intent_value.value,
                status=status,
                payload=payload,
                result=result,
                error=result.get("message") if result.get("error") else None,
            )
            return result
        if intent_value == Intent.DELETE_NUTRITION:
            result = _delete_nutrition_logs(payload, client=health_client)
            status = "error" if result.get("error") else "success"
            record_health_action(
                intent_value.value,
                status=status,
                payload=payload,
                result=result,
                error=result.get("message") if result.get("error") else None,
            )
            return result
        if intent_value == Intent.UPDATE_EXERCISE:
            result = _update_exercise_log(payload, client=health_client, user_text=user_text)
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value in {Intent.LOG_NUTRITION, Intent.LOG_HYDRATION, Intent.LOG_WEIGHT}:
            result = _create_data_point_with_retry(
                intent_value,
                payload,
                client=health_client,
                user_text=user_text,
            )
            if intent_value == Intent.LOG_WEIGHT and result and not result.get("error"):
                grams = payload.get("weight_grams")
                if grams:
                    from ..services.weight_tracking import record_weight_after_sync

                    record_weight_after_sync(
                        weight_kg=float(grams) / 1000.0,
                        logged_at_hkt=payload.get("logged_at_hkt"),
                        notes=payload.get("notes"),
                        google_health_resource=_extract_resource_name(result),
                        source="whatsapp",
                    )
            record_health_action(intent_value.value, status="success", payload=payload, result=result)
            return result
        if intent_value == Intent.LOG_EXERCISE:
            result = _create_exercise_logs_with_retry(
                payload,
                client=health_client,
                user_text=user_text,
            )
            status = "error" if result.get("error") else "success"
            record_health_action(
                intent_value.value,
                status=status,
                payload=payload,
                result=result,
                error=result.get("message") if result.get("error") else None,
            )
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
    except GoogleAuthRequiredError:
        if sender_phone:
            from ..integrations.google_auth import notify_google_auth_required

            notify_google_auth_required(sender_phone)
        raise
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
