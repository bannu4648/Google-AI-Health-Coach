"""Tests for dual-model LLM routing."""

from __future__ import annotations

import pytest

from backend.health_coach.integrations.llm.routing import DualModelLLMProvider


class _FakeProvider:
    def __init__(self, name: str, model: str):
        self._name = name
        self._model = model
        self.calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def rate_limit_user_reply(self) -> str:
        return f"{self._name} limited"

    def generate_json(self, **kwargs):
        self.calls.append(kwargs.get("purpose", ""))
        if kwargs.get("images"):
            return {"provider": self._name, "vision": True}
        return {"provider": self._name, "vision": False}

    def transcribe_audio(self, *, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        self.calls.append("transcribe_audio")
        return "hello"

    def summarize_document(self, **kwargs) -> str:
        self.calls.append("summarize_document")
        return "summary"


def test_dual_model_routes_text_to_text_provider():
    vision = _FakeProvider("gemini", "gemini-2.5-flash")
    text = _FakeProvider("glm", "@cf/zai-org/glm-5.2")
    provider = DualModelLLMProvider(vision=vision, text=text)

    result = provider.generate_json(
        purpose="route_message",
        system_prompt="sys",
        user_prompt="user",
    )

    assert result == {"provider": "glm", "vision": False}
    assert text.calls == ["route_message"]
    assert vision.calls == []
    assert provider.provider_name == "glm"


def test_dual_model_routes_images_to_vision_provider():
    vision = _FakeProvider("gemini", "gemini-2.5-flash")
    text = _FakeProvider("glm", "@cf/zai-org/glm-5.2")
    provider = DualModelLLMProvider(vision=vision, text=text)

    result = provider.generate_json(
        purpose="analyze_food_image",
        system_prompt="sys",
        user_prompt="user",
        images=[(b"img", "image/jpeg")],
    )

    assert result == {"provider": "gemini", "vision": True}
    assert vision.calls == ["analyze_food_image"]
    assert text.calls == []
    assert provider.provider_name == "gemini"


def test_dual_model_routes_summarize_health_to_vision_provider():
    vision = _FakeProvider("gemini", "gemini-2.5-flash")
    text = _FakeProvider("glm", "@cf/zai-org/glm-5.2")
    provider = DualModelLLMProvider(vision=vision, text=text)

    result = provider.generate_json(
        purpose="summarize_health_data",
        system_prompt="sys",
        user_prompt="user",
    )

    assert result == {"provider": "gemini", "vision": False}
    assert vision.calls == ["summarize_health_data"]
    assert text.calls == []
    assert provider.provider_name == "gemini"


def test_dual_model_delegates_multimodal_helpers():
    vision = _FakeProvider("gemini", "gemini-2.5-flash")
    text = _FakeProvider("glm", "@cf/zai-org/glm-5.2")
    provider = DualModelLLMProvider(vision=vision, text=text)

    assert provider.transcribe_audio(audio_bytes=b"audio") == "hello"
    assert provider.summarize_document(
        document_bytes=b"doc",
        mime_type="application/pdf",
        system_prompt="sys",
        user_prompt="user",
    ) == "summary"
    assert vision.calls == ["transcribe_audio", "summarize_document"]
