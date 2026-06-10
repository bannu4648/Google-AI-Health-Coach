"""
SQLite persistence for the local health coach.

This module intentionally uses the Python standard library so the app can stay
small and fully local. All payloads are stored as JSON text with secrets
redacted before persistence.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(os.getenv("HEALTH_COACH_DB_PATH", DATA_DIR / "health_coach.sqlite3"))
if not DB_PATH.is_absolute():
    DB_PATH = PROJECT_ROOT / DB_PATH

_LOCK = threading.RLock()

SECRET_KEYS = {
    "authorization",
    "access_token",
    "token",
    "refresh_token",
    "client_secret",
    "mistral_api_key",
    "whatsapp_access_token",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any) -> str:
    return str(value)


def redact(value: Any) -> Any:
    """Recursively redact likely secret values before logging or storage."""
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in SECRET_KEYS or "token" in key.lower() or "secret" in key.lower():
                clean[key] = "[REDACTED]"
            else:
                clean[key] = redact(item)
        return clean
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def to_json(value: Any) -> str:
    return json.dumps(redact(value), default=_json_default, ensure_ascii=False)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT,
                summary TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                direction TEXT NOT NULL,
                phone TEXT,
                text TEXT,
                status TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                purpose TEXT NOT NULL,
                model TEXT,
                status TEXT,
                latency_ms INTEGER,
                prompt_json TEXT NOT NULL DEFAULT '{}',
                response_json TEXT NOT NULL DEFAULT '{}',
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS google_health_calls (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                method TEXT NOT NULL,
                url TEXT NOT NULL,
                data_type TEXT,
                status_code INTEGER,
                latency_ms INTEGER,
                request_json TEXT NOT NULL DEFAULT '{}',
                response_json TEXT NOT NULL DEFAULT '{}',
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS tavily_calls (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                query TEXT NOT NULL,
                food_display_name TEXT,
                portion_description TEXT,
                status TEXT NOT NULL,
                latency_ms INTEGER,
                result_count INTEGER,
                request_json TEXT NOT NULL DEFAULT '{}',
                response_json TEXT NOT NULL DEFAULT '{}',
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS health_actions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                intent TEXT NOT NULL,
                status TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS job_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                job_name TEXT NOT NULL,
                status TEXT,
                started_at TEXT,
                finished_at TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS coach_notes (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                category TEXT NOT NULL,
                note TEXT NOT NULL,
                source TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS daily_summaries (
                id TEXT PRIMARY KEY,
                date_hkt TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                summary_type TEXT NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                message TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_llm_created ON llm_calls(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_google_created ON google_health_calls(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_tavily_created ON tavily_calls(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_actions_created ON health_actions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON job_runs(created_at DESC);
            """
        )


def _insert(table: str, values: dict[str, Any]) -> str:
    init_db()
    row_id = values.setdefault("id", str(uuid.uuid4()))
    values.setdefault("created_at", utc_now_iso())
    columns = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)
    with connect() as conn:
        conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
    return row_id


def record_event(event_type: str, source: str, *, status: str = "", summary: str = "", payload: Any = None) -> str:
    return _insert(
        "events",
        {
            "event_type": event_type,
            "source": source,
            "status": status,
            "summary": summary,
            "payload_json": to_json(payload or {}),
        },
    )


def record_message(direction: str, *, phone: str | None, text: str | None, status: str = "", payload: Any = None) -> str:
    return _insert(
        "messages",
        {
            "direction": direction,
            "phone": phone,
            "text": text,
            "status": status,
            "payload_json": to_json(payload or {}),
        },
    )


def record_llm_call(
    *,
    purpose: str,
    model: str,
    status: str,
    latency_ms: int | None = None,
    prompt: Any = None,
    response: Any = None,
    error: str | None = None,
) -> str:
    return _insert(
        "llm_calls",
        {
            "purpose": purpose,
            "model": model,
            "status": status,
            "latency_ms": latency_ms,
            "prompt_json": to_json(prompt or {}),
            "response_json": to_json(response or {}),
            "error": error,
        },
    )


def record_tavily_call(
    *,
    query: str,
    status: str,
    food_display_name: str | None = None,
    portion_description: str | None = None,
    latency_ms: int | None = None,
    result_count: int | None = None,
    request: Any = None,
    response: Any = None,
    error: str | None = None,
) -> str:
    return _insert(
        "tavily_calls",
        {
            "query": query,
            "food_display_name": food_display_name,
            "portion_description": portion_description,
            "status": status,
            "latency_ms": latency_ms,
            "result_count": result_count,
            "request_json": to_json(request or {}),
            "response_json": to_json(response or {}),
            "error": error,
        },
    )


def record_google_health_call(
    *,
    method: str,
    url: str,
    data_type: str | None = None,
    status_code: int | None = None,
    latency_ms: int | None = None,
    request: Any = None,
    response: Any = None,
    error: str | None = None,
) -> str:
    return _insert(
        "google_health_calls",
        {
            "method": method,
            "url": url,
            "data_type": data_type,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "request_json": to_json(request or {}),
            "response_json": to_json(response or {}),
            "error": error,
        },
    )


def record_health_action(intent: str, *, status: str, payload: Any = None, result: Any = None, error: str | None = None) -> str:
    return _insert(
        "health_actions",
        {
            "intent": intent,
            "status": status,
            "payload_json": to_json(payload or {}),
            "result_json": to_json(result or {}),
            "error": error,
        },
    )


def record_job_run(
    *,
    job_name: str,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    payload: Any = None,
    result: Any = None,
    error: str | None = None,
) -> str:
    return _insert(
        "job_runs",
        {
            "job_name": job_name,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "payload_json": to_json(payload or {}),
            "result_json": to_json(result or {}),
            "error": error,
        },
    )


def upsert_daily_summary(date_hkt: str, *, summary_type: str, metrics: Any, message: str) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO daily_summaries (id, date_hkt, created_at, summary_type, metrics_json, message)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date_hkt) DO UPDATE SET
                created_at = excluded.created_at,
                summary_type = excluded.summary_type,
                metrics_json = excluded.metrics_json,
                message = excluded.message
            """,
            (str(uuid.uuid4()), date_hkt, utc_now_iso(), summary_type, to_json(metrics), message),
        )


def add_coach_note(category: str, note: str, *, source: str = "system", payload: Any = None) -> str:
    now = utc_now_iso()
    return _insert(
        "coach_notes",
        {
            "created_at": now,
            "updated_at": now,
            "category": category,
            "note": note,
            "source": source,
            "payload_json": to_json(payload or {}),
        },
    )


def fetch_recent(table: str, *, limit: int = 50) -> list[dict[str, Any]]:
    allowed = {
        "events",
        "messages",
        "llm_calls",
        "google_health_calls",
        "tavily_calls",
        "health_actions",
        "job_runs",
        "coach_notes",
        "daily_summaries",
    }
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    init_db()
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_recent_messages_for_phone(phone: str, *, limit: int = 16) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT direction, text, created_at, status
            FROM messages
            WHERE phone = ? AND text IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (phone, limit),
        ).fetchall()
    return list(reversed([dict(row) for row in rows]))


init_db()
