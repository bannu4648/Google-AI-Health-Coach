"""
Provider-agnostic LLM interface for the health coach.

All coach agents (router, vision, nutrition, research, summarizer) depend on this
protocol — not on a specific vendor SDK.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal surface required by AIEngine and VisionAgent."""

    @property
    def provider_name(self) -> str:
        """Short id, e.g. gemini, mistral, openai."""

    @property
    def model_name(self) -> str:
        """Model id sent to record_llm_call."""

    @property
    def rate_limit_user_reply(self) -> str:
        """WhatsApp message when rate limits are exhausted after retries."""

    def generate_json(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        images: list[tuple[bytes, str]] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the model."""

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
        """Parse model JSON into a Pydantic response model."""

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        """True for HTTP 429 / quota errors from this provider."""
