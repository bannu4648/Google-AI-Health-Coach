"""Dashboard analytics helpers backed by SQLite."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .database import fetch_recent
from .timezone import local_date_str


def parse_json_field(row: dict[str, Any], field: str) -> Any:
    value = row.get(field)
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def hydrate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hydrated = []
    for row in rows:
        item = dict(row)
        for field in list(item):
            if field.endswith("_json"):
                item[field[:-5]] = parse_json_field(item, field)
        hydrated.append(item)
    return hydrated


def recent_table(table: str, *, limit: int = 50) -> list[dict[str, Any]]:
    return hydrate_rows(fetch_recent(table, limit=limit))


def overview() -> dict[str, Any]:
    return technical_summary()


def technical_summary() -> dict[str, Any]:
    messages = recent_table("messages", limit=200)
    llm_calls = recent_table("llm_calls", limit=200)
    google_calls = recent_table("google_health_calls", limit=200)
    tavily_calls = recent_table("tavily_calls", limit=200)
    actions = recent_table("health_actions", limit=200)
    jobs = recent_table("job_runs", limit=50)
    summaries = recent_table("daily_summaries", limit=7)

    return {
        "date_hkt": local_date_str(),
        "counts": {
            "messages": len(messages),
            "llm_calls": len(llm_calls),
            "google_health_calls": len(google_calls),
            "tavily_calls": len(tavily_calls),
            "health_actions": len(actions),
            "job_runs": len(jobs),
        },
        "message_status": dict(Counter(row.get("status") or "unknown" for row in messages)),
        "llm_status": dict(Counter(row.get("status") or "unknown" for row in llm_calls)),
        "google_status": dict(Counter(str(row.get("status_code") or "error") for row in google_calls)),
        "tavily_status": dict(Counter(row.get("status") or "unknown" for row in tavily_calls)),
        "action_status": dict(Counter(row.get("status") or "unknown" for row in actions)),
        "latest_summary": summaries[0] if summaries else None,
        "latest_messages": messages[:10],
        "latest_google_calls": google_calls[:10],
    }


def _latest_daily_summary() -> dict[str, Any] | None:
    summaries = recent_table("daily_summaries", limit=1)
    return summaries[0] if summaries else None


def _metric_value(snapshot: dict[str, Any], path: list[str], default: Any = 0) -> Any:
    value: Any = snapshot
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _sanitize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "date_hkt": snapshot.get("date_hkt") or local_date_str(),
        "readiness": snapshot.get("readiness", {"score": None, "label": "unknown", "reasons": []}),
        "recommendations": snapshot.get("recommendations", []),
        "metrics": {
            "steps": _metric_value(snapshot, ["steps", "count"]),
            "active_zone_minutes": _metric_value(snapshot, ["active_zone_minutes", "raw", "minutesSum"]),
            "workouts": _metric_value(snapshot, ["exercise", "count"]),
            "sleep_sessions": _metric_value(snapshot, ["sleep", "count"]),
            "meals_logged": _metric_value(snapshot, ["nutrition", "count"]),
            "hydration_ml": _metric_value(snapshot, ["hydration", "raw", "volumeMilliliters"]),
            "heart_rate_samples": _metric_value(snapshot, ["heart_rate", "sample_count"]),
        },
    }


def health_overview() -> dict[str, Any]:
    from ..services.coach_state import build_coach_state_snapshot
    from ..services.coaching_preferences import get_coaching_focus
    from ..services.scheduler import get_scheduler_config

    summary = _latest_daily_summary()
    snapshot = summary.get("metrics", {}) if summary else {}
    actions = recent_table("health_actions", limit=10)
    messages = recent_table("messages", limit=10)
    notes = recent_table("coach_notes", limit=5)
    coach_state = build_coach_state_snapshot()
    scheduler = get_scheduler_config()
    return {
        **_sanitize_snapshot(snapshot),
        "coach_message": summary.get("message") if summary else "",
        "summary_type": summary.get("summary_type") if summary else None,
        "coaching_panel": {
            "coaching_focus": get_coaching_focus(),
            "goals": coach_state.get("goals", []),
            "plan_adherence": coach_state.get("plan_adherence"),
            "plan_summary": coach_state.get("plan_summary", ""),
            "next_nudges": {
                "morning_summary": scheduler.get("morning_summary_time"),
                "evening_summary": scheduler.get("evening_summary_time"),
                "readiness_nudge": scheduler.get("readiness_nudge_time") or None,
                "workout_nudge": scheduler.get("workout_nudge_time"),
                "weekly_recap": f"{scheduler.get('weekly_recap_day')} {scheduler.get('weekly_recap_time')}",
            },
            "scheduler_enabled": scheduler.get("enabled") == "true",
        },
        "recent_activity": [
            {
                "created_at": row.get("created_at"),
                "intent": row.get("intent"),
                "status": row.get("status"),
            }
            for row in actions[:6]
        ],
        "recent_messages": [
            {
                "created_at": row.get("created_at"),
                "direction": row.get("direction"),
                "text": row.get("text"),
                "status": row.get("status"),
            }
            for row in messages[:5]
        ],
        "coach_notes": [
            {
                "created_at": row.get("created_at"),
                "category": row.get("category"),
                "note": row.get("note"),
            }
            for row in notes
        ],
    }


def health_trends(*, days: int = 14) -> dict[str, Any]:
    summaries = list(reversed(recent_table("daily_summaries", limit=days)))
    items = []
    for row in summaries:
        snapshot = row.get("metrics", {})
        readiness = snapshot.get("readiness", {})
        metrics = _sanitize_snapshot(snapshot).get("metrics", {})
        items.append(
            {
                "date_hkt": row.get("date_hkt") or snapshot.get("date_hkt"),
                "readiness_score": readiness.get("score"),
                "steps": metrics.get("steps", 0),
                "active_zone_minutes": metrics.get("active_zone_minutes", 0),
                "workouts": metrics.get("workouts", 0),
                "meals_logged": metrics.get("meals_logged", 0),
                "sleep_sessions": metrics.get("sleep_sessions", 0),
            }
        )
    return {"items": items}


def metric_ranges() -> dict[str, Any]:
    """Return chart-friendly rows from cached call/action history."""
    google_calls = recent_table("google_health_calls", limit=500)
    actions = recent_table("health_actions", limit=500)
    return {
        "google_health_calls": [
            {
                "created_at": row.get("created_at"),
                "data_type": row.get("data_type"),
                "status_code": row.get("status_code"),
                "latency_ms": row.get("latency_ms"),
            }
            for row in google_calls
        ],
        "health_actions": [
            {
                "created_at": row.get("created_at"),
                "intent": row.get("intent"),
                "status": row.get("status"),
            }
            for row in actions
        ],
    }
