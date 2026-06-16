"""User goals storage and progress tracking (local SQLite)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..core.database import connect, init_db, utc_now_iso


def _row_to_goal(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["target"] = json.loads(item.pop("target_json") or "{}")
    item["progress"] = json.loads(item.pop("progress_json") or "{}")
    return item


def log_goal(
    *,
    category: str,
    goal_text: str,
    target: dict[str, Any] | None = None,
    deadline_hkt: str | None = None,
    google_health_sync: str = "none",
) -> dict[str, Any]:
    init_db()
    now = utc_now_iso()
    goal_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO user_goals
            (id, created_at, updated_at, category, goal_text, target_json, progress_json,
             deadline_hkt, status, google_health_sync)
            VALUES (?, ?, ?, ?, ?, ?, '{}', ?, 'active', ?)
            """,
            (
                goal_id,
                now,
                now,
                category,
                goal_text,
                json.dumps(target or {}, ensure_ascii=False),
                deadline_hkt,
                google_health_sync,
            ),
        )
        row = conn.execute("SELECT * FROM user_goals WHERE id = ?", (goal_id,)).fetchone()
    return _row_to_goal(row)


def update_goal(
    goal_id: str | None = None,
    *,
    goal_text: str | None = None,
    target: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    status: str | None = None,
    deadline_hkt: str | None = None,
) -> dict[str, Any] | None:
    init_db()
    now = utc_now_iso()
    with connect() as conn:
        if goal_id:
            row = conn.execute("SELECT * FROM user_goals WHERE id = ?", (goal_id,)).fetchone()
        elif goal_text:
            row = conn.execute(
                """
                SELECT * FROM user_goals
                WHERE status = 'active' AND goal_text LIKE ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (f"%{goal_text}%",),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM user_goals WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None

        current = _row_to_goal(row)
        merged_target = {**current.get("target", {}), **(target or {})}
        merged_progress = {**current.get("progress", {}), **(progress or {})}
        conn.execute(
            """
            UPDATE user_goals
            SET updated_at = ?, goal_text = ?, target_json = ?, progress_json = ?,
                status = COALESCE(?, status), deadline_hkt = COALESCE(?, deadline_hkt)
            WHERE id = ?
            """,
            (
                now,
                goal_text or current["goal_text"],
                json.dumps(merged_target, ensure_ascii=False),
                json.dumps(merged_progress, ensure_ascii=False),
                status,
                deadline_hkt,
                current["id"],
            ),
        )
        updated = conn.execute("SELECT * FROM user_goals WHERE id = ?", (current["id"],)).fetchone()
    return _row_to_goal(updated) if updated else None


def fetch_active_goals(*, limit: int = 10) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM user_goals
            WHERE status = 'active'
              AND goal_text NOT LIKE '%help log%'
              AND goal_text NOT LIKE '%log my goals%'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_goal(row) for row in rows]


def fetch_all_goals(*, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM user_goals WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM user_goals ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_goal(row) for row in rows]


def format_goals_for_reply(goals: list[dict[str, Any]]) -> str:
    if not goals:
        return "You don't have any goals saved yet. Tell me what you're working toward."
    lines = ["*Your goals*"]
    for goal in goals:
        progress = goal.get("progress") or {}
        progress_note = ""
        if progress.get("sessions_completed") is not None:
            target_sessions = (goal.get("target") or {}).get("sessions_per_week")
            if target_sessions:
                progress_note = f" ({progress['sessions_completed']}/{target_sessions} this week)"
        status = goal.get("status", "active")
        lines.append(f"• [{status}] {goal.get('goal_text', '')}{progress_note}")
    return "\n".join(lines)


def sync_fitness_plan_goal(*, goals: dict[str, Any], week_start_hkt: str) -> dict[str, Any] | None:
    """Create or update a fitness goal when a weekly plan is saved."""
    sessions = goals.get("gym_sessions") or goals.get("sessions_per_week")
    if not sessions:
        for value in goals.values():
            if isinstance(value, str) and "gym" in value.lower():
                sessions = 2
                break
    if not sessions:
        return None

    goal_text = goals.get("focus") or goals.get("summary") or f"Gym {sessions}x per week"
    if not isinstance(goal_text, str):
        goal_text = f"Gym {sessions}x per week"

    existing = None
    for goal in fetch_active_goals(limit=20):
        if goal.get("category") == "fitness" and "gym" in goal.get("goal_text", "").lower():
            existing = goal
            break

    target = {"sessions_per_week": int(sessions), "week_start_hkt": week_start_hkt}
    if existing:
        return update_goal(existing["id"], target=target, progress={"sessions_completed": 0})
    return log_goal(
        category="fitness",
        goal_text=str(goal_text),
        target=target,
        google_health_sync="exercise",
    )


def increment_workout_goal_progress() -> None:
    for goal in fetch_active_goals(limit=20):
        if goal.get("category") != "fitness":
            continue
        target = goal.get("target") or {}
        if not target.get("sessions_per_week"):
            continue
        progress = goal.get("progress") or {}
        completed = int(progress.get("sessions_completed", 0)) + 1
        update_goal(goal["id"], progress={"sessions_completed": completed})
