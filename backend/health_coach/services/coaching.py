"""Premium-coach inspired analysis and proactive message generation."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any

from ..agent.engine import AIEngine
from ..core.database import add_coach_note, upsert_daily_summary
from ..core.timezone import (
    enrich_health_api_result_for_llm,
    format_utc_iso,
    get_user_tz,
    local_date_str,
    now_local,
)
from ..integrations.google_health import GoogleHealthAPIError, GoogleHealthClient

WEEKLY_LOOKBACK_DAYS = 7


def local_day_bounds_utc(day: datetime | None = None) -> tuple[str, str]:
    local = (day or now_local()).astimezone(get_user_tz())
    start_local = datetime.combine(local.date(), time.min, tzinfo=get_user_tz())
    end_local = start_local + timedelta(days=1)
    return format_utc_iso(start_local), format_utc_iso(end_local)


def last_night_sleep_bounds_utc() -> tuple[str, str]:
    """Sleep window: yesterday 18:00 HKT through today noon (or now if earlier)."""
    local = now_local()
    yesterday = local.date() - timedelta(days=1)
    start_local = datetime.combine(yesterday, time(18, 0), tzinfo=get_user_tz())
    noon_today = datetime.combine(local.date(), time(12, 0), tzinfo=get_user_tz())
    end_local = min(local, noon_today)
    if end_local <= start_local:
        end_local = local
    return format_utc_iso(start_local), format_utc_iso(end_local)


def week_bounds_utc() -> tuple[str, str]:
    """Rolling 7-day window ending at the start of tomorrow in HKT."""
    local = now_local()
    end_local = datetime.combine(local.date(), time.min, tzinfo=get_user_tz()) + timedelta(days=1)
    start_local = end_local - timedelta(days=WEEKLY_LOOKBACK_DAYS)
    return format_utc_iso(start_local), format_utc_iso(end_local)


def _safe_call(default: Any, fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except (GoogleHealthAPIError, ValueError, KeyError):
        return default


def _latest_rollup_value(result: dict[str, Any], key: str) -> dict[str, Any]:
    points = result.get("rollupDataPoints", [])
    for point in points:
        if key in point:
            return point[key]
    return {}


def _rollup_daily_series(
    result: dict[str, Any],
    metric_key: str,
    value_field: str,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for point in result.get("rollupDataPoints", []):
        block = point.get(metric_key, {})
        civil = point.get("startTime", {}).get("civilDateTime", {})
        date_label = "-".join(
            str(civil.get(part, "")).zfill(2)
            for part in ("year", "month", "day")
            if civil.get(part) is not None
        )
        series.append(
            {
                "date_hkt": date_label or None,
                "value": block.get(value_field, 0) or 0,
            }
        )
    return series


def _count_data_points_by_day(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        interval = item.get("interval", {})
        start = interval.get("startTime", {})
        civil = start.get("civilDateTime", start.get("utcDateTime", {}))
        if not civil:
            continue
        day = "-".join(
            str(civil.get(part, "")).zfill(2)
            for part in ("year", "month", "day")
            if civil.get(part) is not None
        )
        if day:
            counts[day] = counts.get(day, 0) + 1
    return counts


def _fetch_range_metrics(
    health: GoogleHealthClient,
    *,
    start: str,
    end: str,
    include_rollups: bool = True,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"range_utc": {"start": start, "end": end}}

    if include_rollups:
        steps_result = _safe_call(
            {},
            health.daily_roll_up,
            "steps",
            start_time=start,
            end_time=end,
        )
        azm_result = _safe_call(
            {},
            health.daily_roll_up,
            "active-zone-minutes",
            start_time=start,
            end_time=end,
        )
        hydration_result = _safe_call(
            {},
            health.daily_roll_up,
            "hydration-log",
            start_time=start,
            end_time=end,
        )
        metrics["steps"] = {
            "count": int(_latest_rollup_value(steps_result, "steps").get("countSum", 0) or 0),
            "raw": steps_result,
        }
        metrics["active_zone_minutes"] = {
            "raw": _latest_rollup_value(azm_result, "activeZoneMinutes"),
        }
        metrics["hydration"] = {
            "raw": _latest_rollup_value(hydration_result, "hydrationLog"),
        }

    exercise_result = _safe_call(
        {},
        health.list_data_points,
        "exercise",
        start_time=start,
        end_time=end,
    )
    sleep_result = _safe_call(
        {},
        health.reconcile_data_points,
        "sleep",
        start_time=start,
        end_time=end,
    )
    heart_result = _safe_call(
        {},
        health.reconcile_data_points,
        "heart-rate",
        start_time=start,
        end_time=end,
        page_size=50,
    )
    rhr_result = _safe_call(
        {},
        health.reconcile_data_points,
        "daily-resting-heart-rate",
        start_time=start,
        end_time=end,
    )
    meals_result = _safe_call(
        {},
        health.list_data_points,
        "nutrition-log",
        start_time=start,
        end_time=end,
        page_size=50,
    )

    exercise_items = exercise_result.get("dataPoints", [])
    sleep_items = sleep_result.get("dataPoints", [])
    meal_items = meals_result.get("dataPoints", [])

    metrics["exercise"] = {"count": len(exercise_items), "items": exercise_items}
    metrics["sleep"] = {"count": len(sleep_items), "items": sleep_items}
    metrics["heart_rate"] = {
        "sample_count": len(heart_result.get("dataPoints", [])),
        "items": heart_result.get("dataPoints", [])[:10],
    }
    metrics["resting_heart_rate"] = rhr_result
    metrics["nutrition"] = {"count": len(meal_items), "items": meal_items}
    return metrics


def get_weekly_health_trends(*, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    health = client or GoogleHealthClient()
    start, end = week_bounds_utc()

    steps_result = _safe_call({}, health.daily_roll_up, "steps", start_time=start, end_time=end)
    azm_result = _safe_call(
        {},
        health.daily_roll_up,
        "active-zone-minutes",
        start_time=start,
        end_time=end,
    )
    hydration_result = _safe_call(
        {},
        health.daily_roll_up,
        "hydration-log",
        start_time=start,
        end_time=end,
    )
    exercise_result = _safe_call(
        {},
        health.list_data_points,
        "exercise",
        start_time=start,
        end_time=end,
    )
    sleep_result = _safe_call(
        {},
        health.reconcile_data_points,
        "sleep",
        start_time=start,
        end_time=end,
    )
    meals_result = _safe_call(
        {},
        health.list_data_points,
        "nutrition-log",
        start_time=start,
        end_time=end,
        page_size=100,
    )

    steps_series = _rollup_daily_series(steps_result, "steps", "countSum")
    azm_series = _rollup_daily_series(azm_result, "activeZoneMinutes", "minutesSum")
    hydration_series = _rollup_daily_series(hydration_result, "hydrationLog", "volumeMilliliters")

    exercise_items = exercise_result.get("dataPoints", [])
    sleep_items = sleep_result.get("dataPoints", [])
    meal_items = meals_result.get("dataPoints", [])

    total_steps = sum(int(row.get("value", 0) or 0) for row in steps_series)
    days_with_steps = len([row for row in steps_series if int(row.get("value", 0) or 0) > 0])
    avg_steps = int(total_steps / days_with_steps) if days_with_steps else 0

    return {
        "days": WEEKLY_LOOKBACK_DAYS,
        "range_utc": {"start": start, "end": end},
        "steps": {
            "daily": steps_series,
            "total": total_steps,
            "average_on_active_days": avg_steps,
        },
        "active_zone_minutes": {"daily": azm_series},
        "hydration_ml": {
            "daily": hydration_series,
            "total": sum(int(row.get("value", 0) or 0) for row in hydration_series),
        },
        "exercise": {
            "total_sessions": len(exercise_items),
            "by_day": _count_data_points_by_day(exercise_items),
        },
        "sleep": {
            "total_sessions": len(sleep_items),
            "by_day": _count_data_points_by_day(sleep_items),
        },
        "nutrition": {
            "total_meals": len(meal_items),
            "by_day": _count_data_points_by_day(meal_items),
        },
    }


def get_daily_health_snapshot(
    *,
    client: GoogleHealthClient | None = None,
    day: datetime | None = None,
) -> dict[str, Any]:
    """Today's health metrics (single calendar day in HKT)."""
    health = client or GoogleHealthClient()
    start, end = local_day_bounds_utc(day)
    snapshot = _fetch_range_metrics(health, start=start, end=end)
    snapshot["date_hkt"] = local_date_str(day)
    snapshot["readiness"] = readiness_score(snapshot)
    snapshot["recommendations"] = recommendations(snapshot)
    return snapshot


def get_evening_health_snapshot(
    *,
    client: GoogleHealthClient | None = None,
    day: datetime | None = None,
) -> dict[str, Any]:
    """Evening recap: activities, meals, and hydration for today only."""
    snapshot = get_daily_health_snapshot(client=client, day=day)
    snapshot["summary_type"] = "evening"
    snapshot["scope"] = "today_only"
    return snapshot


def get_morning_health_snapshot(*, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    """Morning briefing: last night's sleep plus weekly trends for today planning."""
    health = client or GoogleHealthClient()
    sleep_start, sleep_end = last_night_sleep_bounds_utc()

    last_night_sleep = _safe_call(
        {},
        health.reconcile_data_points,
        "sleep",
        start_time=sleep_start,
        end_time=sleep_end,
    )
    sleep_items = last_night_sleep.get("dataPoints", [])

    weekly = get_weekly_health_trends(client=health)
    today = get_daily_health_snapshot(client=health)

    snapshot = {
        "summary_type": "morning",
        "scope": "last_night_sleep_and_weekly_trends",
        "date_hkt": local_date_str(),
        "last_night_sleep": {
            "range_utc": {"start": sleep_start, "end": sleep_end},
            "count": len(sleep_items),
            "items": sleep_items,
        },
        "weekly_trends": weekly,
        "today_so_far": {
            "steps": today.get("steps", {}).get("count", 0),
            "exercise_count": today.get("exercise", {}).get("count", 0),
            "meals_logged": today.get("nutrition", {}).get("count", 0),
        },
    }
    snapshot["readiness"] = morning_readiness_score(snapshot)
    snapshot["recommendations"] = morning_recommendations(snapshot)
    return snapshot


def readiness_score(snapshot: dict[str, Any]) -> dict[str, Any]:
    score = 70
    reasons: list[str] = []

    steps = snapshot.get("steps", {}).get("count", 0)
    exercises = snapshot.get("exercise", {}).get("count", 0)
    sleep_count = snapshot.get("sleep", {}).get("count", 0)

    if sleep_count == 0:
        score -= 10
        reasons.append("No sleep data is available yet, so recovery confidence is lower.")
    else:
        score += 10
        reasons.append("Sleep data is available, so recovery guidance can be more personalized.")

    if steps >= 10000:
        score += 8
        reasons.append("You reached a strong step volume today.")
    elif steps < 4000:
        score -= 8
        reasons.append("Step volume is on the lighter side today.")

    if exercises:
        score += 6
        reasons.append("A workout was logged today.")

    score = max(0, min(100, score))
    if score >= 80:
        label = "ready"
    elif score >= 55:
        label = "steady"
    else:
        label = "recover"
    return {"score": score, "label": label, "reasons": reasons}


def morning_readiness_score(snapshot: dict[str, Any]) -> dict[str, Any]:
    score = 70
    reasons: list[str] = []

    sleep_count = snapshot.get("last_night_sleep", {}).get("count", 0)
    weekly = snapshot.get("weekly_trends", {})
    avg_steps = weekly.get("steps", {}).get("average_on_active_days", 0)
    workout_total = weekly.get("exercise", {}).get("total_sessions", 0)

    if sleep_count == 0:
        score -= 12
        reasons.append("No sleep data from last night yet — recovery guidance is less certain.")
    else:
        score += 12
        reasons.append("Last night's sleep data is available for recovery planning.")

    if avg_steps >= 8000:
        score += 6
        reasons.append("Weekly step volume has been solid.")
    elif avg_steps and avg_steps < 5000:
        score -= 6
        reasons.append("Weekly step volume has been on the lighter side.")

    if workout_total >= 3:
        score += 4
        reasons.append("You have been active across the week.")
    elif workout_total == 0:
        score -= 4
        reasons.append("No workouts logged this week yet.")

    score = max(0, min(100, score))
    if score >= 80:
        label = "ready"
    elif score >= 55:
        label = "steady"
    else:
        label = "recover"
    return {"score": score, "label": label, "reasons": reasons}


def recommendations(snapshot: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    steps = snapshot.get("steps", {}).get("count", 0)
    readiness = snapshot.get("readiness", {}).get("label")
    meals = snapshot.get("nutrition", {}).get("count", 0)

    if readiness == "recover":
        recs.append("Keep tomorrow lighter: mobility, an easy walk, and earlier sleep.")
    elif readiness == "ready":
        recs.append("You can handle a more focused workout tomorrow if your schedule allows.")
    else:
        recs.append("Aim for a balanced day tomorrow: moderate movement and consistent meals.")

    if steps < 7000:
        recs.append("Add a 20-30 minute walk to lift baseline activity.")
    if meals == 0:
        recs.append("Log meals as you go so nutrition feedback becomes more useful.")
    return recs


def morning_recommendations(snapshot: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    readiness = snapshot.get("readiness", {}).get("label")
    weekly = snapshot.get("weekly_trends", {})
    avg_steps = weekly.get("steps", {}).get("average_on_active_days", 0)
    meals_by_day = weekly.get("nutrition", {}).get("by_day", {})
    sleep_count = snapshot.get("last_night_sleep", {}).get("count", 0)

    if readiness == "recover":
        recs.append("Plan a lighter day: easy movement, hydration, and an earlier wind-down tonight.")
    elif readiness == "ready":
        recs.append("Good window for a focused workout or longer walk today.")
    else:
        recs.append("Aim for steady movement today and consistent meal timing.")

    if sleep_count == 0:
        recs.append("Check that sleep tracking synced — last night has no sleep session yet.")
    if avg_steps and avg_steps < 7000:
        recs.append("Weekly steps are below target — add a 20-30 minute walk today.")
    if len(meals_by_day) < 4:
        recs.append("Meal logging has been sparse this week — log today's meals for better nutrition feedback.")
    return recs


def build_daily_coach_message(summary_type: str, snapshot: dict[str, Any]) -> str:
    if summary_type == "evening":
        draft = (
            f"Evening recap for {snapshot['date_hkt']}: "
            f"{snapshot['steps']['count']} steps, {snapshot['exercise']['count']} workouts, "
            f"{snapshot['nutrition']['count']} meals logged today. "
            f"Readiness is {snapshot['readiness']['score']}/100 ({snapshot['readiness']['label']})."
        )
        user_text = (
            "Create an evening recap for TODAY only. Summarize today's steps, workouts, "
            "meals, hydration, and heart-rate activity. Reflect on what went well and "
            "give practical sleep-prep advice for tonight. Do not discuss multi-day trends."
        )
    else:
        weekly = snapshot.get("weekly_trends", {})
        last_night = snapshot.get("last_night_sleep", {})
        draft = (
            f"Morning briefing for {snapshot['date_hkt']}: "
            f"last night {last_night.get('count', 0)} sleep session(s); "
            f"7-day avg steps {weekly.get('steps', {}).get('average_on_active_days', 0)}; "
            f"{weekly.get('exercise', {}).get('total_sessions', 0)} workouts this week. "
            f"Readiness is {snapshot['readiness']['score']}/100 ({snapshot['readiness']['label']})."
        )
        user_text = (
            "Create a morning proactive health coach message. Lead with last night's sleep "
            "(duration/quality if available). Then summarize the past week's activity, sleep, "
            "meals, and hydration trends. End with clear, actionable suggestions for what to "
            "focus on TODAY."
        )

    llm_payload = enrich_health_api_result_for_llm(snapshot)
    try:
        engine = AIEngine()
        return engine.summarize_health_data(
            user_text=user_text,
            draft_reply=draft,
            api_result=llm_payload,
        )
    except Exception:
        return draft + " " + " ".join(snapshot.get("recommendations", [])[:2])


def create_daily_summary(summary_type: str, *, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    if summary_type == "evening":
        snapshot = get_evening_health_snapshot(client=client)
    else:
        snapshot = get_morning_health_snapshot(client=client)

    message = build_daily_coach_message(summary_type, snapshot)
    upsert_daily_summary(
        snapshot.get("date_hkt") or local_date_str(),
        summary_type=summary_type,
        metrics=snapshot,
        message=message,
    )
    add_coach_note(
        "daily_summary",
        f"{summary_type}: readiness {snapshot['readiness']['score']} - {snapshot['readiness']['label']}",
        source="scheduler",
        payload=snapshot,
    )
    return {"snapshot": snapshot, "message": message}
