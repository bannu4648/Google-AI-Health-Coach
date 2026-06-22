import pytest

from backend.health_coach.integrations.llm import create_llm_provider
from backend.health_coach.integrations.llm.gemini import GeminiProvider


def test_create_llm_provider_defaults_to_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    provider = create_llm_provider(api_key="test-key")
    assert isinstance(provider, GeminiProvider)
    assert provider.provider_name == "gemini"


def test_create_llm_provider_rejects_unknown(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_provider("not-a-real-provider", api_key="x")
