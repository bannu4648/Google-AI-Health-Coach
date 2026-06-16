from unittest.mock import MagicMock

from backend.health_coach.agent.engine import AIEngine
from backend.health_coach.integrations.llm.gemini import GeminiProvider, RATE_LIMIT_USER_REPLY


class FakeRateLimitError(Exception):
    status_code = 429


def test_gemini_is_rate_limit_error_detects_status_code():
    assert GeminiProvider.is_rate_limit_error(FakeRateLimitError("rate limited"))


def test_gemini_is_rate_limit_error_detects_message():
    assert GeminiProvider.is_rate_limit_error(RuntimeError("HTTP 429 Too Many Requests"))


def test_route_message_returns_rate_limit_reply():
    llm = MagicMock()
    llm.model_name = "gemini-2.0-flash"
    llm.rate_limit_user_reply = RATE_LIMIT_USER_REPLY
    llm.is_rate_limit_error = GeminiProvider.is_rate_limit_error
    llm.generate_json.side_effect = FakeRateLimitError("rate limited")

    engine = AIEngine(llm=llm)
    routed = engine.route_message("summarize my heart rate")
    assert routed.conversational_reply == RATE_LIMIT_USER_REPLY
