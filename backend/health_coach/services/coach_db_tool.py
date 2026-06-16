"""Read-only text2sql access to coach SQLite with hard guardrails."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from typing import Any, Callable

from ..core.database import connect, init_db, utc_now_iso

ALLOWED_TABLES = frozenset(
    {
        "fitness_plans",
        "fitness_workouts",
        "mood_logs",
        "cycle_logs",
        "user_goals",
        "coach_notes",
        "daily_summaries",
        "health_actions",
        "v_active_fitness_plan",
        "v_recent_mood",
        "v_active_goals",
    }
)

BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|PRAGMA|CREATE|REPLACE|TRUNCATE|VACUUM)\b",
    re.IGNORECASE,
)

DEFAULT_LIMIT = 50
MAX_LIMIT = 100
MAX_ROWS_FOR_LLM = 20
QUERY_TIMEOUT_SECONDS = 2.0


def _normalize_sql(sql: str) -> str:
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].strip()
    if ";" in stripped:
        raise ValueError("Multiple SQL statements are not allowed.")
    return stripped


def _extract_tables(sql: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE):
        found.add(match.group(1).lower())
    return found


def _ensure_limit(sql: str, *, limit: int = DEFAULT_LIMIT) -> str:
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
        if limit_match and int(limit_match.group(1)) > MAX_LIMIT:
            sql = re.sub(r"\bLIMIT\s+\d+", f"LIMIT {MAX_LIMIT}", sql, count=1, flags=re.IGNORECASE)
        return sql
    return f"{sql} LIMIT {limit}"


def _truncate_row(row: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > 500:
            clean[key] = value[:500] + "…"
        else:
            clean[key] = value
    return clean


def _record_query(
    *,
    sql_text: str,
    purpose: str,
    status: str,
    row_count: int | None = None,
    error: str | None = None,
    result: list[dict[str, Any]] | None = None,
) -> str:
    query_id = str(uuid.uuid4())
    init_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO coach_db_queries
            (id, created_at, purpose, sql_text, status, row_count, error, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                utc_now_iso(),
                purpose,
                sql_text,
                status,
                row_count,
                error,
                json.dumps(result or [], ensure_ascii=False, default=str),
            ),
        )
    return query_id


def query_coach_db(sql: str, *, purpose: str = "") -> dict[str, Any]:
    """Execute a validated read-only SELECT against coach SQLite."""
    try:
        normalized = _normalize_sql(sql)
        if not normalized.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")

        if BLOCKED_KEYWORDS.search(normalized):
            raise ValueError("Query contains blocked keywords.")

        tables = _extract_tables(normalized)
        if not tables:
            raise ValueError("Could not determine table(s) from query.")
        disallowed = tables - ALLOWED_TABLES
        if disallowed:
            raise ValueError(f"Table(s) not allowed: {', '.join(sorted(disallowed))}")

        limited_sql = _ensure_limit(normalized)

        init_db()
        with connect() as conn:
            conn.execute(f"PRAGMA busy_timeout = {int(QUERY_TIMEOUT_SECONDS * 1000)}")
            cursor = conn.execute(limited_sql)
            columns = [description[0] for description in cursor.description or []]
            raw_rows = cursor.fetchall()

        rows = [_truncate_row(dict(zip(columns, row))) for row in raw_rows[:MAX_ROWS_FOR_LLM]]
        _record_query(
            sql_text=limited_sql,
            purpose=purpose,
            status="success",
            row_count=len(raw_rows),
            result=rows,
        )
        return {
            "rows": rows,
            "row_count": len(raw_rows),
            "truncated_for_llm": len(raw_rows) > MAX_ROWS_FOR_LLM,
            "sql_executed": limited_sql,
        }
    except (sqlite3.Error, ValueError) as exc:
        _record_query(
            sql_text=sql,
            purpose=purpose,
            status="error",
            error=str(exc),
        )
        return {"error": True, "message": str(exc), "rows": []}


def lookup_coach_data(
    *,
    natural_question: str,
    sql_query: str = "",
    generate_sql: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """
    Hybrid coach memory lookup: generate SQL (if needed), run query_coach_db, retry once on error.
    """
    sql = (sql_query or "").strip()
    if not sql:
        if generate_sql is None:
            return {
                "error": True,
                "message": "No SQL query provided and no SQL generator configured.",
                "rows": [],
            }
        generated = generate_sql(natural_question=natural_question)
        sql = getattr(generated, "sql_query", "") or (generated.get("sql_query") if isinstance(generated, dict) else "")
        if not sql:
            return {"error": True, "message": "Could not generate a coach DB query.", "rows": []}

    result = query_coach_db(sql, purpose=natural_question[:200])
    if not result.get("error"):
        result["natural_question"] = natural_question
        result["sql_generated"] = not bool((sql_query or "").strip())
        return result

    if generate_sql is None:
        return result

    fixed = generate_sql(
        natural_question=natural_question,
        error_hint=result.get("message", ""),
        previous_sql=sql,
    )
    retry_sql = getattr(fixed, "sql_query", "") or (fixed.get("sql_query") if isinstance(fixed, dict) else "")
    if not retry_sql or retry_sql.strip() == sql.strip():
        return result

    retry = query_coach_db(retry_sql, purpose=natural_question[:200])
    retry["natural_question"] = natural_question
    retry["sql_retried"] = True
    return retry
