from backend.health_coach.agent.intent_registry import is_batch_nutrition as _is_batch_nutrition
from backend.health_coach.core.payloads import expand_nutrition_items


def test_is_batch_nutrition_true_for_items():
    payload = {
        "items": [
            {"food_display_name": "a"},
            {"food_display_name": "b"},
        ]
    }
    assert _is_batch_nutrition(payload) is True
    assert len(expand_nutrition_items(payload)) == 2


def test_is_batch_nutrition_false_for_single():
    assert _is_batch_nutrition({"food_display_name": "rice"}) is False
