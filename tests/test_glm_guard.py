"""Tests for GLM cache guard and daily usage tracking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.health_coach.integrations.llm.glm_guard import GuardedGLMProvider, _GLMCacheStore


class _FakeGLM:
    provider_name = "glm"
    model_name = "@cf/zai-org/glm-5.2"
    rate_limit_user_reply = "rate limited"
    calls = 0

    def generate_json(self, **kwargs):
        self.calls += 1
        return {"ok": True, "purpose": kwargs.get("purpose")}


@pytest.fixture
def cache_store(tmp_path):
    return _GLMCacheStore(db_path=tmp_path / "glm_cache.sqlite3")


def test_guarded_glm_cache_hit_skips_api(cache_store):
    inner = _FakeGLM()
    guarded = GuardedGLMProvider(inner, cache_store=cache_store, cache_enabled=True)

    first = guarded.generate_json(
        purpose="route_message",
        system_prompt="sys",
        user_prompt="user",
    )
    second = guarded.generate_json(
        purpose="route_message",
        system_prompt="sys",
        user_prompt="user",
    )

    assert first == second == {"ok": True, "purpose": "route_message"}
    assert inner.calls == 1


def test_guarded_glm_distinct_prompts_miss_cache(cache_store):
    inner = _FakeGLM()
    guarded = GuardedGLMProvider(inner, cache_store=cache_store, cache_enabled=True)

    guarded.generate_json(purpose="a", system_prompt="sys", user_prompt="one")
    guarded.generate_json(purpose="b", system_prompt="sys", user_prompt="two")

    assert inner.calls == 2


def test_guarded_glm_daily_soft_limit_warning(cache_store, caplog):
    inner = _FakeGLM()
    guarded = GuardedGLMProvider(
        inner,
        cache_store=cache_store,
        cache_enabled=False,
        daily_soft_limit=2,
    )

    with caplog.at_level("WARNING"):
        guarded.generate_json(purpose="a", system_prompt="s", user_prompt="1")
        guarded.generate_json(purpose="b", system_prompt="s", user_prompt="2")
        guarded.generate_json(purpose="c", system_prompt="s", user_prompt="3")

    assert inner.calls == 3
    assert any("GLM daily soft limit reached" in rec.message for rec in caplog.records)
