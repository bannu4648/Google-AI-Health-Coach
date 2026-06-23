"""Day-review routing and OAuth notify helpers."""

from backend.health_coach.agent.graph import (
    _apply_day_review_guard,
    _is_day_review_request,
)
from backend.health_coach.agent.engine import Intent


def test_day_review_request_detection():
    text = (
        "get me last logged food and exercises from yday and tell me if it was "
        "a healthy day towards my goal or not"
    )
    assert _is_day_review_request(text)


def test_apply_day_review_guard_yesterday():
    intent, payload = _apply_day_review_guard(
        "food and exercises from yday — healthy for my goals?",
        Intent.COACHING_CHAT.value,
    )
    assert intent == Intent.EVALUATE_DAY.value
    assert payload["day_offset_days"] == -1


def test_apply_day_review_guard_ignores_unrelated():
    intent, payload = _apply_day_review_guard("log 2 eggs for breakfast", Intent.LOG_NUTRITION.value)
    assert intent == Intent.LOG_NUTRITION.value
    assert payload == {}
