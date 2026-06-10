"""Premium-coach inspired analysis and proactive message generation."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any

from ..agent.engine import AIEngine
from ..core.database import add_coach_note, upsert_daily_summary
from ..core.timezone import format_utc_iso, get_user_tz, local_date_str, now_local
from ..integrations.google_health import GoogleHealthAPIError, GoogleHealthClient


def local_day_bounds_utc(day: datetime | None = None) -> tuple[str, str]:
    local = (day or now_local()).astimezone(get_user_tz())
    start_local = datetime.combine(local.date(), time.min, tzinfo=get_user_tz())
    end_local = start_local + timedelta(days=1)
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


def get_daily_health_snapshot(
    *,
    client: GoogleHealthClient | None = None,
    day: datetime | None = None,
) -> dict[str, Any]:
    health = client or GoogleHealthClient()
    start, end = local_day_bounds_utc(day)

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
        page_size=25,
    )
    hydration_result = _safe_call(
        {},
        health.daily_roll_up,
        "hydration-log",
        start_time=start,
        end_time=end,
    )

    steps = _latest_rollup_value(steps_result, "steps")
    active_zone_minutes = _latest_rollup_value(azm_result, "activeZoneMinutes")
    hydration = _latest_rollup_value(hydration_result, "hydrationLog")

    snapshot = {
        "date_hkt": local_date_str(day),
        "range_utc": {"start": start, "end": end},
        "steps": {
            "count": int(steps.get("countSum", 0) or 0),
            "raw": steps_result,
        },
        "active_zone_minutes": {
            "raw": active_zone_minutes,
        },
        "exercise": {
            "count": len(exercise_result.get("dataPoints", [])),
            "items": exercise_result.get("dataPoints", []),
        },
        "sleep": {
            "count": len(sleep_result.get("dataPoints", [])),
            "items": sleep_result.get("dataPoints", []),
        },
        "heart_rate": {
            "sample_count": len(heart_result.get("dataPoints", [])),
            "items": heart_result.get("dataPoints", [])[:10],
        },
        "resting_heart_rate": rhr_result,
        "nutrition": {
            "count": len(meals_result.get("dataPoints", [])),
            "items": meals_result.get("dataPoints", []),
        },
        "hydration": {
            "raw": hydration,
        },
    }
    snapshot["readiness"] = readiness_score(snapshot)
    snapshot["recommendations"] = recommendations(snapshot)
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


def build_daily_coach_message(summary_type: str, snapshot: dict[str, Any]) -> str:
    draft = (
        f"{summary_type.title()} check-in for {snapshot['date_hkt']}: "
        f"{snapshot['steps']['count']} steps, {snapshot['exercise']['count']} workouts, "
        f"{snapshot['nutrition']['count']} meals logged. "
        f"Readiness is {snapshot['readiness']['score']}/100 ({snapshot['readiness']['label']})."
    )
    try:
        engine = AIEngine()
        return engine.summarize_health_data(
            user_text=f"Create a {summary_type} proactive health coach message.",
            draft_reply=draft,
            api_result=snapshot,
        )
    except Exception:
        return draft + " " + " ".join(snapshot.get("recommendations", [])[:2])


def create_daily_summary(summary_type: str, *, client: GoogleHealthClient | None = None) -> dict[str, Any]:
    snapshot = get_daily_health_snapshot(client=client)
    message = build_daily_coach_message(summary_type, snapshot)
    upsert_daily_summary(
        snapshot["date_hkt"],
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
