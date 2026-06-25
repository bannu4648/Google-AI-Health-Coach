import pytest

from backend.health_coach.integrations.llm import create_llm_provider
from backend.health_coach.integrations.llm.gemini import GeminiProvider
from backend.health_coach.integrations.llm.routing import DualModelLLMProvider


def test_create_llm_provider_defaults_to_gemini(monkeypatch):
    monkeypatch.setenv("LLM_ROUTING_MODE", "all_google")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    provider = create_llm_provider(api_key="test-key")
    assert isinstance(provider, GeminiProvider)
    assert provider.provider_name == "gemini"


def test_create_llm_provider_gemini_glm_mode(monkeypatch):
    monkeypatch.setenv("LLM_ROUTING_MODE", "gemini_glm")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)

    provider = create_llm_provider()
    assert isinstance(provider, DualModelLLMProvider)
    assert provider.vision.provider_name == "gemini"
    assert provider.text.provider_name == "glm"


def test_create_llm_provider_gemini_mistral_mode(monkeypatch):
    monkeypatch.setenv("LLM_ROUTING_MODE", "gemini_mistral")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)

    provider = create_llm_provider()
    assert isinstance(provider, DualModelLLMProvider)
    assert provider.vision.provider_name == "gemini"
    assert provider.text.provider_name == "mistral"


def test_create_llm_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_ROUTING_MODE", "all_google")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_provider("not-a-real-provider", api_key="x")
