"""Task-aware dual-model LLM routing (vision vs text)."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .protocol import LLMProvider

logger = logging.getLogger(__name__)


class DualModelLLMProvider:
    """
    Route multimodal / vision calls to a vision-capable provider (Gemini)
    and all text/JSON reasoning to a separate text provider (GLM or Mistral).
    """

    def __init__(self, *, vision: LLMProvider, text: LLMProvider):
        self._vision = vision
        self._text = text
        self._last_used: LLMProvider = text

    @property
    def vision(self) -> LLMProvider:
        return self._vision

    @property
    def text(self) -> LLMProvider:
        return self._text

    @property
    def provider_name(self) -> str:
        return self._last_used.provider_name

    @property
    def model_name(self) -> str:
        return self._last_used.model_name

    @property
    def rate_limit_user_reply(self) -> str:
        return self._text.rate_limit_user_reply

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        from .gemini import GeminiProvider
        from .glm import GLMProvider
        from .mistral import MistralProvider

        for provider_cls in (GeminiProvider, GLMProvider, MistralProvider):
            checker = getattr(provider_cls, "is_rate_limit_error", None)
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
            self._last_used = self._vision
            logger.debug("DualModel routing %s to vision provider (%s)", purpose, self._vision.provider_name)
            return self._vision.generate_json(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=images,
            )

        self._last_used = self._text
        logger.debug("DualModel routing %s to text provider (%s)", purpose, self._text.provider_name)
        return self._text.generate_json(
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

    def transcribe_audio(self, *, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        transcribe = getattr(self._vision, "transcribe_audio", None)
        if not callable(transcribe):
            return ""
        self._last_used = self._vision
        return transcribe(audio_bytes=audio_bytes, mime_type=mime_type) or ""

    def summarize_document(
        self,
        *,
        document_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        summarize = getattr(self._vision, "summarize_document", None)
        if not callable(summarize):
            return ""
        self._last_used = self._vision
        return summarize(
            document_bytes=document_bytes,
            mime_type=mime_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ) or ""
