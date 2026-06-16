"""Mistral LLM provider (optional — requires `pip install mistralai`)."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from .rate_limit import RateLimitedLLMProvider

load_dotenv()

DEFAULT_MODEL = os.getenv("MISTRAL_MODEL", os.getenv("LLM_MODEL", "mistral-large-latest"))
DEFAULT_API_KEY = os.getenv("MISTRAL_API_KEY", os.getenv("LLM_API_KEY", ""))
DEFAULT_CALL_DELAY = float(
    os.getenv("MISTRAL_CALL_DELAY_SECONDS", os.getenv("LLM_CALL_DELAY_SECONDS", "2"))
)
DEFAULT_MAX_RETRIES = int(
    os.getenv("MISTRAL_RATE_LIMIT_MAX_RETRIES", os.getenv("LLM_RATE_LIMIT_MAX_RETRIES", "3"))
)
DEFAULT_BACKOFF = float(
    os.getenv(
        "MISTRAL_RATE_LIMIT_BACKOFF_SECONDS",
        os.getenv("LLM_RATE_LIMIT_BACKOFF_SECONDS", "2"),
    )
)


class MistralProvider(RateLimitedLLMProvider):
    """Mistral JSON provider. Vision is not supported — use Gemini for food photos."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        call_delay_seconds: float | None = None,
        rate_limit_max_retries: int | None = None,
        rate_limit_backoff_seconds: float | None = None,
    ):
        try:
            from mistralai.client import Mistral
        except ImportError as exc:
            raise ImportError(
                "Mistral provider requires mistralai. Install with: pip install mistralai"
            ) from exc

        key = api_key or DEFAULT_API_KEY
        if not key:
            raise ValueError("Set MISTRAL_API_KEY or LLM_API_KEY in your .env file.")

        self._client = Mistral(api_key=key)
        super().__init__(
            provider_name="mistral",
            model_name=model_name,
            call_delay_seconds=DEFAULT_CALL_DELAY if call_delay_seconds is None else call_delay_seconds,
            rate_limit_max_retries=DEFAULT_MAX_RETRIES
            if rate_limit_max_retries is None
            else rate_limit_max_retries,
            rate_limit_backoff_seconds=DEFAULT_BACKOFF
            if rate_limit_backoff_seconds is None
            else rate_limit_backoff_seconds,
            rate_limit_user_reply=(
                "The LLM API rate limit was hit — I've queued a short pause and will be ready "
                "again in a few seconds. Please resend your message."
            ),
        )

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        if "429" in message or "rate limit" in message:
            return True
        status_code = getattr(exc, "status_code", None)
        return status_code == 429

    def _complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        images: list[tuple[bytes, str]] | None,
    ) -> dict[str, Any]:
        if images:
            raise NotImplementedError(
                "Mistral provider does not support vision in this project yet. "
                "Set LLM_PROVIDER=gemini for food photos."
            )
        response = self._client.chat.complete(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
