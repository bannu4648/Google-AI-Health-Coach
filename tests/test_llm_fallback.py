"""Tests for Gemini → Mistral (or other) LLM failover."""

from __future__ import annotations

import pytest

from backend.health_coach.integrations.llm.fallback import FallbackLLMProvider
from backend.health_coach.integrations.llm.factory import create_llm_provider


class _FakeProvider:
    def __init__(self, name: str, model: str, *, fail: bool = False, fail_on_images: bool = False):
        self._name = name
        self._model = model
        self._fail = fail
        self._fail_on_images = fail_on_images
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def rate_limit_user_reply(self) -> str:
        return f"{self._name} rate limited"

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        return "429" in str(exc)

    def generate_json(self, **kwargs):
        self.calls += 1
        images = kwargs.get("images")
        if images and self._fail_on_images:
            raise RuntimeError("vision not supported")
        if self._fail:
            raise RuntimeError("429 rate limit exceeded")
        return {"ok": True, "provider": self._name}


def test_fallback_uses_secondary_on_primary_rate_limit():
    primary = _FakeProvider("gemini", "gemini-2.5-flash", fail=True)
    fallback = _FakeProvider("mistral", "mistral-large-latest")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    result = provider.generate_json(
        purpose="route_message",
        system_prompt="sys",
        user_prompt="user",
    )

    assert result["provider"] == "mistral"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert provider.model_name == "mistral-large-latest"


def test_fallback_does_not_use_secondary_for_vision():
    primary = _FakeProvider("gemini", "gemini-2.5-flash", fail_on_images=True)
    fallback = _FakeProvider("mistral", "mistral-large-latest")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    with pytest.raises(RuntimeError, match="vision not supported"):
        provider.generate_json(
            purpose="analyze_food_image",
            system_prompt="sys",
            user_prompt="user",
            images=[(b"img", "image/jpeg")],
        )

    assert primary.calls == 1
    assert fallback.calls == 0


def test_create_llm_provider_wraps_fallback_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_ROUTING_MODE", "all_google")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral")

    provider = create_llm_provider()
    assert isinstance(provider, FallbackLLMProvider)
    assert provider.primary.provider_name == "gemini"
    assert provider.fallback.provider_name == "mistral"
