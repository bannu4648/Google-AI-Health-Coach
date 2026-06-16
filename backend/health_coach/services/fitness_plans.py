"""Weekly fitness plan storage and retrieval (local SQLite until Google Fitness tab API)."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from ..core.database import connect, init_db, utc_now_iso
from ..core.timezone import get_user_tz, now_local

DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

DAY_NAME_TO_INDEX: dict[str, int] = {}
for _index, _name in enumerate(DAY_NAMES):
    DAY_NAME_TO_INDEX[_name.lower()] = _index
    DAY_NAME_TO_INDEX[_name.lower()[:3]] = _index


def _week_start_hkt(day: datetime | None = None) -> str:
    local = (day or now_local()).astimezone(get_user_tz())
    monday = local.date() - timedelta(days=local.weekday())
    return monday.isoformat()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _plan_week_end(week_start_hkt: str) -> date:
    return _parse_date(week_start_hkt) + timedelta(days=6)


def _load_plan_row(plan_row: Any) -> dict[str, Any]:
    result = dict(plan_row)
    result["goals"] = json.loads(result.pop("goals_json") or "{}")
    result["weekly_targets"] = json.loads(result.pop("weekly_targets_json") or "{}")
    with connect() as conn:
        workouts = conn.execute(
            "SELECT * FROM fitness_workouts WHERE plan_id = ? ORDER BY day_of_week, created_at",
            (result["id"],),
        ).fetchall()
    result["workouts"] = []
    for row in workouts:
        item = dict(row)
        item["steps"] = json.loads(item.pop("steps_json") or "[]")
        result["workouts"].append(item)
    return result


def get_relevant_active_plan(*, for_date: datetime | None = None) -> dict[str, Any] | None:
    """
    Find the best active fitness plan for a given date.

    Priority: plan week containing for_date, else nearest upcoming week, else latest active.
    """
    init_db()
    local = (for_date or now_local()).astimezone(get_user_tz())
    target = local.date()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM fitness_plans
            WHERE status = 'active'
            ORDER BY week_start_hkt DESC, created_at DESC
            """
        ).fetchall()
    if not rows:
        return None

    containing: list[dict[str, Any]] = []
    upcoming: list[dict[str, Any]] = []
    for row in rows:
        week_start = _parse_date(row["week_start_hkt"])
        week_end = week_start + timedelta(days=6)
        if week_start <= target <= week_end:
            containing.append(dict(row))
        elif week_start >= target:
            upcoming.append(dict(row))

    if containing:
        return _load_plan_row(containing[0])
    if upcoming:
        return _load_plan_row(min(upcoming, key=lambda item: item["week_start_hkt"]))
    return _load_plan_row(dict(rows[0]))


def save_fitness_plan(
    *,
    week_start_hkt: str | None,
    goals: dict[str, Any],
    weekly_targets: dict[str, Any],
    workouts: list[dict[str, Any]],
) -> dict[str, Any]:
    init_db()
    week = week_start_hkt or _week_start_hkt()
    now = utc_now_iso()
    plan_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "UPDATE fitness_plans SET status = 'archived' WHERE week_start_hkt = ? AND status = 'active'",
            (week,),
        )
        conn.execute(
            """
            INSERT INTO fitness_plans (id, created_at, updated_at, week_start_hkt, goals_json, weekly_targets_json, status)
            VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                plan_id,
                now,
                now,
                week,
                json.dumps(goals, ensure_ascii=False),
                json.dumps(weekly_targets, ensure_ascii=False),
            ),
        )
        for workout in workouts:
            conn.execute(
                """
                INSERT INTO fitness_workouts
                (id, plan_id, created_at, day_of_week, title, exercise_type, steps_json, duration_minutes, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    str(uuid.uuid4()),
                    plan_id,
                    now,
                    int(workout.get("day_of_week", 0)),
                    workout.get("title") or "Workout",
                    workout.get("exercise_type"),
                    json.dumps(workout.get("steps") or [], ensure_ascii=False),
                    workout.get("duration_minutes"),
                ),
            )
    return get_active_plan(week_start_hkt=week) or get_relevant_active_plan() or {
        "plan_id": plan_id,
        "week_start_hkt": week,
    }


def get_active_plan(*, week_start_hkt: str | None = None) -> dict[str, Any] | None:
    if week_start_hkt:
        init_db()
        with connect() as conn:
            plan = conn.execute(
                """
                SELECT * FROM fitness_plans
                WHERE week_start_hkt = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
                """,
                (week_start_hkt,),
            ).fetchone()
        if not plan:
            return None
        return _load_plan_row(plan)
    return get_relevant_active_plan()


def parse_day_filter(day_filter: Any) -> list[int]:
    """Parse router day_filter into weekday indices (0=Monday)."""
    if day_filter is None:
        return []
    if isinstance(day_filter, int):
        return [day_filter]
    if isinstance(day_filter, str):
        parts = [part.strip().lower() for part in day_filter.replace("/", ",").split(",") if part.strip()]
        indices: list[int] = []
        for part in parts:
            if part.isdigit():
                indices.append(int(part))
            elif part in DAY_NAME_TO_INDEX:
                indices.append(DAY_NAME_TO_INDEX[part])
        return indices
    if isinstance(day_filter, list):
        indices: list[int] = []
        for item in day_filter:
            indices.extend(parse_day_filter(item))
        return indices
    return []


def plan_adherence_summary(plan: dict[str, Any], *, for_date: datetime | None = None) -> dict[str, Any]:
    workouts = plan.get("workouts") or []
    completed = [workout for workout in workouts if workout.get("completed_at")]
    local = (for_date or now_local()).astimezone(get_user_tz())
    target = local.date()
    week_start = _parse_date(plan["week_start_hkt"])
    week_end = week_start + timedelta(days=6)
    base = {
        "total_workouts": len(workouts),
        "completed_workouts": len(completed),
        "week_start_hkt": plan.get("week_start_hkt"),
    }
    if target < week_start:
        base["status"] = "upcoming"
        base["label"] = f"Plan starts {plan.get('week_start_hkt')} (not started yet)"
        base["completed_workouts"] = 0
        return base
    if target > week_end:
        base["status"] = "past"
        base["label"] = f"Plan week ended {week_end.isoformat()}"
        return base
    base["status"] = "active"
    base["label"] = f"{len(completed)}/{len(workouts)} workouts done this plan week"
    return base


def get_todays_workout(*, day: datetime | None = None) -> dict[str, Any] | None:
    plan = get_relevant_active_plan(for_date=day)
    if not plan:
        return None
    weekday = (day or now_local()).weekday()
    for workout in plan.get("workouts", []):
        if int(workout.get("day_of_week", -1)) == weekday and not workout.get("completed_at"):
            return {**workout, "plan_id": plan["id"], "week_start_hkt": plan["week_start_hkt"]}
    return None


def complete_workout(workout_id: str) -> dict[str, Any] | None:
    init_db()
    now = utc_now_iso()
    with connect() as conn:
        conn.execute(
            "UPDATE fitness_workouts SET completed_at = ? WHERE id = ?",
            (now, workout_id),
        )
        row = conn.execute(
            "SELECT * FROM fitness_workouts WHERE id = ?",
            (workout_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["steps"] = json.loads(item.pop("steps_json") or "[]")
    return item


def format_workout_for_reply(workout: dict[str, Any], *, day_label: str | None = None) -> str:
    title = workout.get("title", "Workout")
    prefix = f"{day_label}: " if day_label else ""
    lines = [f"{prefix}*{title}*"]
    if workout.get("duration_minutes"):
        lines.append(f"~{workout['duration_minutes']} min")
    if workout.get("completed_at"):
        lines.append("(completed)")
    steps = workout.get("steps") or []
    for index, step in enumerate(steps, start=1):
        text = str(step).strip()
        if text and text[0].isdigit() and "." in text[:4]:
            lines.append(text)
        else:
            lines.append(f"{index}. {text}")
    return "\n".join(lines)


def format_full_plan_for_reply(plan: dict[str, Any], *, day_indices: list[int] | None = None) -> str:
    """Format Mon–Sun plan with steps for WhatsApp (max ~4000 chars)."""
    week = plan.get("week_start_hkt", "")
    adherence = plan_adherence_summary(plan)
    header = (
        f"Week of {week} "
        f"({adherence['completed_workouts']}/{adherence['total_workouts']} workouts done)"
    )
    blocks: list[str] = [header]
    workouts = plan.get("workouts") or []
    if day_indices:
        workouts = [workout for workout in workouts if int(workout.get("day_of_week", -1)) in day_indices]
    for workout in workouts:
        day_index = int(workout.get("day_of_week", 0))
        day_label = DAY_NAMES[day_index] if 0 <= day_index < 7 else f"Day {day_index}"
        blocks.append(format_workout_for_reply(workout, day_label=day_label))
    text = "\n\n".join(blocks).strip()
    if len(text) > 4000:
        return text[:3990] + "\n…(truncated)"
    return text
