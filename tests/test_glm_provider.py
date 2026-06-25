"""Tests for GLM-5.2 Cloudflare Workers AI provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.health_coach.integrations.llm.glm import GLMProvider, _extract_response_text, _parse_json_text


@pytest.fixture
def glm_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-test")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token-test")
    monkeypatch.setenv("GLM_MODEL", "@cf/zai-org/glm-5.2")


def test_glm_provider_parses_run_response_shape(glm_env):
    provider = GLMProvider()
    payload = {"success": True, "result": {"response": '{"intent": "ok"}'}}
    raw = _extract_response_text(payload)
    assert _parse_json_text(raw) == {"intent": "ok"}


def test_glm_provider_parses_choices_shape(glm_env):
    payload = {
        "success": True,
        "result": {"choices": [{"message": {"content": '{"foo": 1}'}}]},
    }
    raw = _extract_response_text(payload)
    assert _parse_json_text(raw) == {"foo": 1}


def test_glm_provider_strips_markdown_fences(glm_env):
    assert _parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}


@patch("backend.health_coach.integrations.llm.glm.requests.post")
def test_glm_provider_posts_to_cloudflare(mock_post, glm_env):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "result": {"response": '{"answer": 42}'},
    }
    mock_post.return_value = mock_response

    provider = GLMProvider(account_id="acct-test", api_key="token-test")
    result = provider.generate_json(
        purpose="test",
        system_prompt="sys",
        user_prompt="user",
        temperature=0.1,
    )

    assert result == {"answer": 42}
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == (
        "https://api.cloudflare.com/client/v4/accounts/acct-test"
        "/ai/run/@cf/zai-org/glm-5.2"
    )
    assert kwargs["headers"]["Authorization"] == "Bearer token-test"
    assert kwargs["json"]["messages"][0]["role"] == "system"
    assert "JSON" in kwargs["json"]["messages"][0]["content"]


def test_glm_provider_rejects_images(glm_env):
    provider = GLMProvider()
    with pytest.raises(NotImplementedError, match="does not support vision"):
        provider.generate_json(
            purpose="vision",
            system_prompt="sys",
            user_prompt="user",
            images=[(b"img", "image/jpeg")],
        )


def test_glm_provider_detects_rate_limit(glm_env):
    assert GLMProvider.is_rate_limit_error(RuntimeError("429 rate limit exceeded"))
    assert GLMProvider.is_rate_limit_error(RuntimeError("quota exceeded"))


def test_glm_provider_requires_credentials(monkeypatch):
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        GLMProvider(account_id="", api_key="token")

    with pytest.raises(ValueError, match="CLOUDFLARE_API_TOKEN"):
        GLMProvider(account_id="acct", api_key="")
