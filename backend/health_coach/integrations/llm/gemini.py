"""Google Gemini LLM provider."""

from __future__ import annotations

import json
import os
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

from .rate_limit import RateLimitedLLMProvider

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", os.getenv("LLM_MODEL", "gemini-2.5-flash"))
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("LLM_API_KEY", ""))
DEFAULT_CALL_DELAY = float(
    os.getenv("GEMINI_CALL_DELAY_SECONDS", os.getenv("LLM_CALL_DELAY_SECONDS", "0"))
)
DEFAULT_MAX_RETRIES = int(
    os.getenv("GEMINI_RATE_LIMIT_MAX_RETRIES", os.getenv("LLM_RATE_LIMIT_MAX_RETRIES", "3"))
)
DEFAULT_BACKOFF = float(
    os.getenv(
        "GEMINI_RATE_LIMIT_BACKOFF_SECONDS",
        os.getenv("LLM_RATE_LIMIT_BACKOFF_SECONDS", "2"),
    )
)

RATE_LIMIT_USER_REPLY = (
    "The LLM API rate limit was hit — I've queued a short pause and will be ready "
    "again in a few seconds. Please resend your message."
)


class GeminiProvider(RateLimitedLLMProvider):
    """Gemini JSON + vision provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        call_delay_seconds: float | None = None,
        rate_limit_max_retries: int | None = None,
        rate_limit_backoff_seconds: float | None = None,
    ):
        key = api_key or DEFAULT_API_KEY
        if not key:
            raise ValueError("Set GEMINI_API_KEY or LLM_API_KEY in your .env file.")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model_name)
        super().__init__(
            provider_name="gemini",
            model_name=model_name,
            call_delay_seconds=DEFAULT_CALL_DELAY if call_delay_seconds is None else call_delay_seconds,
            rate_limit_max_retries=DEFAULT_MAX_RETRIES
            if rate_limit_max_retries is None
            else rate_limit_max_retries,
            rate_limit_backoff_seconds=DEFAULT_BACKOFF
            if rate_limit_backoff_seconds is None
            else rate_limit_backoff_seconds,
            rate_limit_user_reply=RATE_LIMIT_USER_REPLY,
        )

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        if isinstance(exc, google_exceptions.ResourceExhausted):
            return True
        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "quota" in message

    def _complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        images: list[tuple[bytes, str]] | None,
    ) -> dict[str, Any]:
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

    def transcribe_audio(self, *, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        response = self._model.generate_content(
            [
                "Transcribe this voice note exactly. Return only the spoken text, no JSON.",
                {"mime_type": mime_type, "data": audio_bytes},
            ],
            generation_config=genai.GenerationConfig(temperature=0.1),
        )
        return (response.text or "").strip()

    def summarize_document(
        self,
        *,
        document_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = self._model.generate_content(
            [
                f"{system_prompt}\n\n{user_prompt}",
                {"mime_type": mime_type, "data": document_bytes},
            ],
            generation_config=genai.GenerationConfig(temperature=0.2),
        )
        return (response.text or "").strip()
