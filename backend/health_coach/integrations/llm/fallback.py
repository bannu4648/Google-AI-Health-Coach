"""Primary LLM with optional fallback provider (e.g. Gemini → Mistral)."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .protocol import LLMProvider

logger = logging.getLogger(__name__)


def _is_fallbackable(exc: BaseException, primary: LLMProvider) -> bool:
    """Whether to retry the same request on the fallback provider."""
    if isinstance(exc, (NotImplementedError, ImportError)):
        return False

    checker = getattr(type(primary), "is_rate_limit_error", None)
    if callable(checker) and checker(exc):
        return True

    try:
        from google.api_core import exceptions as google_exceptions

        if isinstance(
            exc,
            (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.DeadlineExceeded,
            ),
        ):
            return True
    except ImportError:
        pass

    message = str(exc).lower()
    transient_markers = (
        "429",
        "rate limit",
        "quota",
        "503",
        "500",
        "unavailable",
        "timeout",
        "deadline",
        "internal error",
        "resource exhausted",
        "overloaded",
    )
    return any(marker in message for marker in transient_markers)


class FallbackLLMProvider:
    """
    Try the primary provider first; on transient failure, use the fallback.

    Vision / image requests always stay on the primary (Mistral cannot analyze photos).
    """

    def __init__(self, *, primary: LLMProvider, fallback: LLMProvider):
        self._primary = primary
        self._fallback = fallback
        self._last_used: LLMProvider = primary

    @property
    def provider_name(self) -> str:
        if self._last_used is self._fallback:
            return self._fallback.provider_name
        return self._primary.provider_name

    @property
    def model_name(self) -> str:
        return self._last_used.model_name

    @property
    def rate_limit_user_reply(self) -> str:
        return self._fallback.rate_limit_user_reply

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def fallback(self) -> LLMProvider:
        return self._fallback

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        from .gemini import GeminiProvider
        from .mistral import MistralProvider

        for provider in (GeminiProvider, MistralProvider):
            checker = getattr(provider, "is_rate_limit_error", None)
            if callable(checker) and checker(exc):
                return True
        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "quota" in message

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
            self._last_used = self._primary
            return self._primary.generate_json(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=images,
            )

        try:
            self._last_used = self._primary
            return self._primary.generate_json(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=None,
            )
        except Exception as exc:
            if not _is_fallbackable(exc, self._primary):
                raise
            logger.warning(
                "Primary LLM (%s) failed for %s (%s); falling back to %s.",
                self._primary.provider_name,
                purpose,
                exc,
                self._fallback.provider_name,
            )
            self._last_used = self._fallback
            return self._fallback.generate_json(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=None,
            )

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
                self._last_used.provider_name,
                purpose,
                exc,
            )
            return None
