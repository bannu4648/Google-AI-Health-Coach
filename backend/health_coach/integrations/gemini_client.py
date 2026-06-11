"""
Google Gemini client for JSON and vision calls (free-tier friendly).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_CALL_DELAY_SECONDS = float(os.getenv("GEMINI_CALL_DELAY_SECONDS", "2"))
GEMINI_RATE_LIMIT_MAX_RETRIES = int(os.getenv("GEMINI_RATE_LIMIT_MAX_RETRIES", "3"))
GEMINI_RATE_LIMIT_BACKOFF_SECONDS = float(
    os.getenv("GEMINI_RATE_LIMIT_BACKOFF_SECONDS", "2")
)

T = TypeVar("T")

RATE_LIMIT_USER_REPLY = (
    "Gemini's API rate limit was hit — I've queued a short pause and will be ready "
    "again in a few seconds. Please resend your message."
)


class GeminiClient:
    """Thin Gemini wrapper with spacing, retries, JSON mode, and optional vision."""

    _last_call_at: float = 0.0
    _call_lock = threading.Lock()

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_MODEL,
        call_delay_seconds: float | None = None,
        rate_limit_max_retries: int | None = None,
        rate_limit_backoff_seconds: float | None = None,
    ):
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError("Set GEMINI_API_KEY in your .env file.")
        genai.configure(api_key=key)
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)
        self._call_delay_seconds = (
            GEMINI_CALL_DELAY_SECONDS if call_delay_seconds is None else call_delay_seconds
        )
        self._rate_limit_max_retries = (
            GEMINI_RATE_LIMIT_MAX_RETRIES
            if rate_limit_max_retries is None
            else rate_limit_max_retries
        )
        self._rate_limit_backoff_seconds = (
            GEMINI_RATE_LIMIT_BACKOFF_SECONDS
            if rate_limit_backoff_seconds is None
            else rate_limit_backoff_seconds
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _is_rate_limit_error(exc: BaseException) -> bool:
        if isinstance(exc, google_exceptions.ResourceExhausted):
            return True
        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "quota" in message

    def _wait_for_call_slot(self) -> None:
        if self._call_delay_seconds <= 0:
            return
        with self._call_lock:
            elapsed = time.monotonic() - GeminiClient._last_call_at
            remaining = self._call_delay_seconds - elapsed
            if remaining > 0:
                logger.info("Waiting %.1fs before Gemini call (rate spacing).", remaining)
                time.sleep(remaining)

    def _mark_call_completed(self) -> None:
        with self._call_lock:
            GeminiClient._last_call_at = time.monotonic()

    def _throttle_after_call(self) -> None:
        if self._call_delay_seconds <= 0:
            return
        logger.info("Throttling %.1fs after Gemini call.", self._call_delay_seconds)
        time.sleep(self._call_delay_seconds)
        self._mark_call_completed()

    def _rate_limited_call(self, purpose: str, call: Callable[[], T]) -> T:
        self._wait_for_call_slot()
        last_exc: BaseException | None = None
        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                result = call()
                self._throttle_after_call()
                return result
            except Exception as exc:
                last_exc = exc
                if self._is_rate_limit_error(exc) and attempt < self._rate_limit_max_retries:
                    wait = self._rate_limit_backoff_seconds * (2**attempt)
                    logger.warning(
                        "Gemini rate limit during %s (attempt %d/%d); retrying in %.1fs.",
                        purpose,
                        attempt + 1,
                        self._rate_limit_max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def generate_json(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        images: list[tuple[bytes, str]] | None = None,
    ) -> dict[str, Any]:
        """Return parsed JSON from Gemini."""

        def _call() -> dict[str, Any]:
            parts: list[Any] = [f"{system_prompt}\n\n{user_prompt}"]
            for image_bytes, mime_type in images or []:
                parts.append({"mime_type": mime_type, "data": image_bytes})
            response = self._model.generate_content(
                parts,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
            raw = response.text or "{}"
            return json.loads(raw)

        return self._rate_limited_call(purpose, _call)

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
        """Parse Gemini JSON into a Pydantic model."""
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
            logger.exception("Failed to parse Gemini structured response for %s: %s", purpose, exc)
            return None
