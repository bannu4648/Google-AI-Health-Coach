"""Premium-coach inspired analysis and proactive message generation."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any

from ..agent.engine import AIEngine
from .user_profile import fetch_user_profile_snapshot, format_user_profile_for_prompt
from ..core.database import add_coach_note, upsert_daily_summary
from ..core.timezone import (
    enrich_health_api_result_for_llm,
    format_utc_iso,
    get_user_tz,
    local_date_str,
    now_local,
)
from ..services.fitness_plans import (
    format_workout_for_reply,
    get_relevant_active_plan,
    get_todays_workout,
    plan_adherence_summary,
)
from ..services.user_goals import fetch_active_goals, format_goals_for_reply
from ..services.goal_progress import format_goal_progress_for_summary
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


def _minutes_between_times(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        from ..core.timezone import parse_to_utc

        seconds = (parse_to_utc(end) - parse_to_utc(start)).total_seconds()
    except (TypeError, ValueError):
        return None
    return int(round(seconds / 60)) if seconds >= 0 else None


def _extract_sleep_metrics(sleep_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate duration and stage minutes from raw sleep data points."""
    total_duration = 0
    rem_minutes = 0
    deep_minutes = 0
    light_minutes = 0
    for point in sleep_items:
        sleep = point.get("sleep", {})
        interval = sleep.get("interval", {})
        duration = _minutes_between_times(interval.get("startTime"), interval.get("endTime"))
        if duration:
            total_duration += duration
        for stage in sleep.get("stages", []):
            stage_type = str(stage.get("type", "")).lower()
            stage_mins = _minutes_between_times(stage.get("startTime"), stage.get("endTime")) or 0
            if "rem" in stage_type:
                rem_minutes += stage_mins
            elif "deep" in stage_type:
                deep_minutes += stage_mins
            elif "light" in stage_type or "core" in stage_type:
                light_minutes += stage_mins
    hours = round(total_duration / 60, 1) if total_duration else None
    return {
        "session_count": len(sleep_items),
        "duration_minutes": total_duration,
        "duration_hours": hours,
        "rem_minutes": rem_minutes,
        "deep_minutes": deep_minutes,
        "light_minutes": light_minutes,
    }


def _extract_resting_hr_bpm(rhr_result: dict[str, Any]) -> int | None:
    points = rhr_result.get("dataPoints", [])
    for point in points:
        rhr = point.get("dailyRestingHeartRate", {})
        bpm = rhr.get("beatsPerMinute")
        if bpm is not None:
            return int(bpm)
    return None


def _extract_active_zone_minutes(azm_raw: dict[str, Any]) -> int:
    if not azm_raw:
        return 0
    return int(azm_raw.get("minutesSum", 0) or azm_raw.get("minutes_sum", 0) or 0)


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

    sleep_metrics = _extract_sleep_metrics(sleep_items)
    rhr_bpm = _extract_resting_hr_bpm(rhr_result)
    azm_minutes = _extract_active_zone_minutes(metrics.get("active_zone_minutes", {}).get("raw", {}))

    metrics["exercise"] = {"count": len(exercise_items), "items": exercise_items}
    metrics["sleep"] = {"count": len(sleep_items), "items": sleep_items, **sleep_metrics}
    metrics["heart_rate"] = {
        "sample_count": len(heart_result.get("dataPoints", [])),
        "items": heart_result.get("dataPoints", [])[:10],
    }
    metrics["resting_heart_rate"] = {"bpm": rhr_bpm, "raw": rhr_result}
    metrics["active_zone_minutes"]["total"] = azm_minutes
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
    sleep_metrics = _extract_sleep_metrics(sleep_items)

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
            **sleep_metrics,
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
    sleep = snapshot.get("sleep", {})
    sleep_hours = sleep.get("duration_hours")
    sleep_count = sleep.get("count", 0)
    azm = snapshot.get("active_zone_minutes", {}).get("total", 0)
    rhr = snapshot.get("resting_heart_rate", {}).get("bpm")
    meals = snapshot.get("nutrition", {}).get("count", 0)

    if sleep_count == 0 or not sleep_hours:
        score -= 12
        reasons.append("No sleep data yet — recovery guidance is less certain.")
    elif sleep_hours < 6:
        score -= 10
        reasons.append(f"Sleep was {sleep_hours}h — consider an earlier wind-down tonight.")
    elif sleep_hours >= 7.5:
        score += 10
        reasons.append(f"Solid sleep at {sleep_hours}h supports recovery.")
    else:
        score += 4
        reasons.append(f"Sleep was {sleep_hours}h — decent but room to improve.")

    deep = sleep.get("deep_minutes", 0)
    if deep and deep >= 60:
        score += 4
        reasons.append(f"Deep sleep was {deep} min — good physical recovery signal.")
    elif deep and deep < 30 and sleep_count:
        score -= 3
        reasons.append("Deep sleep was light — keep today moderate.")

    if azm >= 30:
        score -= 4
        reasons.append(f"Active zone minutes are high today ({azm} min) — factor in fatigue.")
    elif azm >= 15:
        score += 2

    if rhr is not None:
        if rhr > 75:
            score -= 4
            reasons.append(f"Resting HR is elevated ({rhr} bpm) — recovery may be incomplete.")
        elif rhr <= 60:
            score += 3

    if steps >= 10000:
        score += 6
        reasons.append(f"Strong step volume today ({steps:,}).")
    elif steps < 4000:
        score -= 6
        reasons.append(f"Steps are light today ({steps:,}).")

    if exercises:
        score += 4
        reasons.append("A workout was logged today.")

    if meals == 0:
        score -= 2
        reasons.append("No meals logged yet — nutrition feedback is limited.")

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

    last_night = snapshot.get("last_night_sleep", {})
    sleep_hours = last_night.get("duration_hours")
    sleep_count = last_night.get("count", 0)
    weekly = snapshot.get("weekly_trends", {})
    avg_steps = weekly.get("steps", {}).get("average_on_active_days", 0)
    workout_total = weekly.get("exercise", {}).get("total_sessions", 0)
    azm_daily = weekly.get("active_zone_minutes", {}).get("daily", [])
    avg_azm = 0
    if azm_daily:
        values = [int(row.get("value", 0) or 0) for row in azm_daily]
        avg_azm = int(sum(values) / len(values)) if values else 0

    if sleep_count == 0 or not sleep_hours:
        score -= 14
        reasons.append("No sleep data from last night — recovery guidance is uncertain.")
    elif sleep_hours < 6:
        score -= 12
        reasons.append(f"Last night was only {sleep_hours}h — plan a lighter day.")
    elif sleep_hours >= 7.5:
        score += 12
        reasons.append(f"Last night's {sleep_hours}h sleep supports a productive day.")
    else:
        score += 5
        reasons.append(f"Last night: {sleep_hours}h sleep — okay, not optimal.")

    rem = last_night.get("rem_minutes", 0)
    if rem and rem >= 90:
        score += 3
        reasons.append(f"REM sleep was {rem} min — cognitive recovery looks good.")
    elif rem and rem < 45 and sleep_count:
        score -= 3
        reasons.append("REM was short — expect lower focus early today.")

    if avg_azm >= 25:
        score -= 4
        reasons.append(f"Weekly strain is elevated (~{avg_azm} AZM/day) — prioritize recovery.")
    elif avg_azm and avg_azm < 10:
        score += 2

    if avg_steps >= 8000:
        score += 6
        reasons.append(f"Weekly steps averaging {avg_steps:,} — solid baseline.")
    elif avg_steps and avg_steps < 5000:
        score -= 6
        reasons.append(f"Weekly steps averaging {avg_steps:,} — below target.")

    if workout_total >= 3:
        score += 4
        reasons.append(f"{workout_total} workouts logged this week.")
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


def _coach_adherence_context() -> dict[str, Any]:
    plan = get_relevant_active_plan()
    goals = fetch_active_goals(limit=3)
    context: dict[str, Any] = {"goals": goals}
    if plan:
        context["plan_adherence"] = plan_adherence_summary(plan)
        context["plan_week"] = plan.get("week_start_hkt")
    return context


def build_daily_coach_message(
    summary_type: str,
    snapshot: dict[str, Any],
    *,
    client: GoogleHealthClient | None = None,
) -> str:
    adherence = _coach_adherence_context()
    plan_line = ""
    if adherence.get("plan_adherence"):
        pa = adherence["plan_adherence"]
        plan_line = f" {pa.get('label', '')}."
    goals_line = ""
    if adherence.get("goals"):
        goals_line = f" Active goals: {len(adherence['goals'])}."

    if summary_type == "evening":
        draft = (
            f"Evening recap for {snapshot['date_hkt']}: "
            f"{snapshot['steps']['count']} steps, {snapshot['exercise']['count']} workouts, "
            f"{snapshot['nutrition']['count']} meals logged today. "
            f"Readiness is {snapshot['readiness']['score']}/100 ({snapshot['readiness']['label']})."
            f"{plan_line}{goals_line}"
        )
        user_text = (
            "Create an evening recap for TODAY only. Summarize today's steps, workouts, "
            "meals, hydration, and heart-rate activity. Reflect on what went well and "
            "give practical sleep-prep advice for tonight. Do not discuss multi-day trends."
        )
        if adherence.get("plan_adherence"):
            pa = adherence["plan_adherence"]
            user_text += f" Fitness plan status: {pa.get('label', '')}."
        if adherence.get("goals"):
            user_text += f" Active goals summary: {format_goals_for_reply(adherence['goals'])}"
        goal_progress = format_goal_progress_for_summary(adherence.get("goals"), snapshot=snapshot, client=client)
        if goal_progress:
            user_text += f" Goal progress today: {goal_progress}."
    else:
        weekly = snapshot.get("weekly_trends", {})
        last_night = snapshot.get("last_night_sleep", {})
        today_workout = get_todays_workout()
        workout_line = ""
        if today_workout:
            workout_line = f" Today's planned workout: {today_workout.get('title')}."
        draft = (
            f"Morning briefing for {snapshot['date_hkt']}: "
            f"last night {last_night.get('count', 0)} sleep session(s); "
            f"7-day avg steps {weekly.get('steps', {}).get('average_on_active_days', 0)}; "
            f"{weekly.get('exercise', {}).get('total_sessions', 0)} workouts this week.{workout_line} "
            f"Readiness is {snapshot['readiness']['score']}/100 ({snapshot['readiness']['label']})."
            f"{plan_line}{goals_line}"
        )
        user_text = (
            "Create a morning proactive health coach message. Lead with last night's sleep "
            "(duration/quality if available). Then summarize the past week's activity, sleep, "
            "meals, and hydration trends."
        )
        if adherence.get("plan_adherence"):
            pa = adherence["plan_adherence"]
            user_text += f" Include fitness plan status: {pa.get('label', '')}."
        if adherence.get("goals"):
            user_text += f" Mention top goals: {format_goals_for_reply(adherence['goals'])}"
        if today_workout:
            user_text += (
                f" Include today's planned workout with steps: {format_workout_for_reply(today_workout)}"
            )
        user_text += " End with clear, actionable suggestions for what to focus on TODAY."

    llm_payload = enrich_health_api_result_for_llm(snapshot)
    llm_payload["coach_adherence"] = adherence
    try:
        engine = AIEngine()
        profile_context = format_user_profile_for_prompt(
            fetch_user_profile_snapshot(client=client)
        )
        return engine.summarize_health_data(
            user_text=user_text,
            draft_reply=draft,
            api_result=llm_payload,
            user_profile_context=profile_context,
        )
    except Exception:
        return draft + " " + " ".join(snapshot.get("recommendations", [])[:2])


def create_daily_summary(summary_type: str, *, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    if summary_type == "evening":
        snapshot = get_evening_health_snapshot(client=client)
    else:
        snapshot = get_morning_health_snapshot(client=client)

    message = build_daily_coach_message(summary_type, snapshot, client=client)
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


def build_weekly_recap_message(*, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    """7-day rollup + LLM-polished recap for Sunday evening."""
    health = client or GoogleHealthClient()
    weekly = get_weekly_health_trends(client=health)
    adherence = _coach_adherence_context()
    goals = fetch_active_goals(limit=5)
    plan = adherence.get("plan_adherence") or {}
    avg_steps = weekly.get("steps", {}).get("average_on_active_days", 0)
    workouts = weekly.get("exercise", {}).get("total_sessions", 0)
    sleep_days = len(weekly.get("sleep", {}).get("by_day", {}))
    meal_days = len(weekly.get("nutrition", {}).get("by_day", {}))
    draft = (
        f"Week in review: avg {avg_steps:,} steps/day, {workouts} workouts, "
        f"sleep logged {sleep_days}/7 nights, meals logged {meal_days}/7 days."
    )
    if plan:
        draft += f" Plan: {plan.get('label', '')}."
    user_text = (
        "Write a warm Sunday evening weekly health recap for WhatsApp. "
        "Include steps average, workouts, sleep and meal logging consistency, "
        "fitness plan adherence, and one focus for next week. Under 900 chars."
    )
    if goals:
        user_text += f" Goals: {format_goals_for_reply(goals)}"
    llm_payload = enrich_health_api_result_for_llm({"weekly_trends": weekly, "coach_adherence": adherence})
    try:
        engine = AIEngine()
        profile_context = format_user_profile_for_prompt(fetch_user_profile_snapshot(client=health))
        message = engine.summarize_health_data(
            user_text=user_text,
            draft_reply=draft,
            api_result=llm_payload,
            user_profile_context=profile_context,
        )
    except Exception:
        message = draft
    add_coach_note("weekly_recap", message[:200], source="scheduler", payload=weekly)
    return {"weekly": weekly, "message": message, "adherence": adherence}


def create_weekly_recap(*, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    return build_weekly_recap_message(client=client)
