"""SQLite-backed response cache and daily usage guard for GLM provider."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from .glm import GLMProvider
from .protocol import LLMProvider

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CACHE_PATH = Path(
    os.getenv("GLM_CACHE_DB_PATH", DATA_DIR / "glm_llm_cache.sqlite3")
)
if not DEFAULT_CACHE_PATH.is_absolute():
    DEFAULT_CACHE_PATH = PROJECT_ROOT / DEFAULT_CACHE_PATH

DEFAULT_TTL_HOURS = float(os.getenv("GLM_CACHE_TTL_HOURS", "24"))
DEFAULT_DAILY_SOFT_LIMIT = int(os.getenv("GLM_DAILY_SOFT_LIMIT", "60"))
CACHE_ENABLED = os.getenv("GLM_CACHE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_LOCK = threading.RLock()
_WARNED_DATES: set[str] = set()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _cache_key(
    *,
    purpose: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    model_name: str,
) -> str:
    payload = "|".join(
        [
            purpose,
            system_prompt,
            user_prompt,
            str(temperature),
            model_name,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _GLMCacheStore:
    def __init__(self, db_path: Path = DEFAULT_CACHE_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with _LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS glm_cache (
                    cache_key TEXT PRIMARY KEY,
                    purpose TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS glm_daily_usage (
                    usage_date TEXT PRIMARY KEY,
                    api_calls INTEGER NOT NULL DEFAULT 0,
                    cache_hits INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def get_cached(self, cache_key: str, *, ttl_hours: float) -> dict[str, Any] | None:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT response_json, created_at FROM glm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None

            created_at = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            if datetime.now(timezone.utc) - created_at > timedelta(hours=ttl_hours):
                conn.execute("DELETE FROM glm_cache WHERE cache_key = ?", (cache_key,))
                return None

            return json.loads(row["response_json"])

    def store_cached(self, cache_key: str, *, purpose: str, response: dict[str, Any]) -> None:
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO glm_cache (cache_key, purpose, response_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    purpose = excluded.purpose,
                    response_json = excluded.response_json,
                    created_at = excluded.created_at
                """,
                (cache_key, purpose, json.dumps(response, ensure_ascii=False), _utc_now_iso()),
            )

    def record_cache_hit(self) -> None:
        today = _utc_today()
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO glm_daily_usage (usage_date, api_calls, cache_hits)
                VALUES (?, 0, 1)
                ON CONFLICT(usage_date) DO UPDATE SET cache_hits = cache_hits + 1
                """,
                (today,),
            )

    def record_api_call(self, *, soft_limit: int) -> None:
        today = _utc_today()
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO glm_daily_usage (usage_date, api_calls, cache_hits)
                VALUES (?, 1, 0)
                ON CONFLICT(usage_date) DO UPDATE SET api_calls = api_calls + 1
                """,
                (today,),
            )
            row = conn.execute(
                "SELECT api_calls FROM glm_daily_usage WHERE usage_date = ?",
                (today,),
            ).fetchone()
            api_calls = int(row["api_calls"]) if row else 0

        if api_calls >= soft_limit and today not in _WARNED_DATES:
            _WARNED_DATES.add(today)
            logger.warning(
                "GLM daily soft limit reached (%d API calls). "
                "Cloudflare Neurons budget may be exhausted.",
                api_calls,
            )


class GuardedGLMProvider:
    """Wraps GLMProvider with response cache and daily usage tracking."""

    def __init__(
        self,
        inner: GLMProvider | None = None,
        *,
        cache_store: _GLMCacheStore | None = None,
        cache_enabled: bool = CACHE_ENABLED,
        ttl_hours: float = DEFAULT_TTL_HOURS,
        daily_soft_limit: int = DEFAULT_DAILY_SOFT_LIMIT,
    ):
        self._inner = inner or GLMProvider()
        self._store = cache_store or _GLMCacheStore()
        self._cache_enabled = cache_enabled
        self._ttl_hours = ttl_hours
        self._daily_soft_limit = daily_soft_limit
        self._last_used = self._inner

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def rate_limit_user_reply(self) -> str:
        return self._inner.rate_limit_user_reply

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        return GLMProvider.is_rate_limit_error(exc)

    def generate_json(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        images: list[tuple[bytes, str]] | None = None,
    ) -> dict[str, Any]:
        if images:
            return self._inner.generate_json(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=images,
            )

        key: str | None = None
        if self._cache_enabled:
            key = _cache_key(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                model_name=self._inner.model_name,
            )
            cached = self._store.get_cached(key, ttl_hours=self._ttl_hours)
            if cached is not None:
                logger.info("GLM cache hit for %s", purpose)
                self._store.record_cache_hit()
                self._last_used = self._inner
                return cached

        result = self._inner.generate_json(
            purpose=purpose,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            images=None,
        )
        self._last_used = self._inner
        self._store.record_api_call(soft_limit=self._daily_soft_limit)

        if self._cache_enabled and key is not None:
            self._store.store_cached(key, purpose=purpose, response=result)

        return result

    def generate_structured(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.2,
        images: list[tuple[bytes, str]] | None = None,
    ) -> BaseModel | None:
        try:
            parsed = self.generate_json(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=images,
            )
            return response_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.exception(
                "Failed to parse %s structured response for %s: %s",
                self._inner.provider_name,
                purpose,
                exc,
            )
            return None
