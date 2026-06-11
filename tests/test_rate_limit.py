from unittest.mock import MagicMock

import pytest

from backend.health_coach.agent.engine import AIEngine, RATE_LIMIT_USER_REPLY


class FakeRateLimitError(Exception):
    status_code = 429


def test_is_rate_limit_error_detects_status_code():
    assert AIEngine._is_rate_limit_error(FakeRateLimitError("rate limited"))


def test_is_rate_limit_error_detects_message():
    assert AIEngine._is_rate_limit_error(RuntimeError("HTTP 429 Too Many Requests"))


def test_rate_limited_call_retries_then_succeeds(monkeypatch):
    engine = AIEngine(api_key="test-key", call_delay_seconds=0)
    calls = {"count": 0}

    def flaky_call():
        calls["count"] += 1
        if calls["count"] < 3:
            raise FakeRateLimitError("rate limited")
        return "ok"

    monkeypatch.setattr(engine, "_wait_for_call_slot", lambda: None)
    monkeypatch.setattr(engine, "_throttle_after_llm_call", lambda: None)
    monkeypatch.setattr("backend.health_coach.agent.engine.time.sleep", lambda _: None)

    assert engine._rate_limited_call("test", flaky_call) == "ok"
    assert calls["count"] == 3


def test_route_message_returns_rate_limit_reply(monkeypatch):
    engine = AIEngine(api_key="test-key", call_delay_seconds=0, rate_limit_max_retries=0)
    engine._client = MagicMock()
    engine._client.generate_json.side_effect = FakeRateLimitError("rate limited")
    engine._client.model_name = "gemini-2.0-flash"

    routed = engine.route_message("summarize my heart rate")
    assert routed.conversational_reply == RATE_LIMIT_USER_REPLY
