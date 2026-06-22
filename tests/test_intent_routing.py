"""Table-driven routing tests for intent capability registry."""

from backend.health_coach.agent.engine import Intent
from backend.health_coach.agent.intent_registry import (
    get_capability,
    is_batch_nutrition,
    route_after_intent,
    route_after_nutrition_lookup,
)


def test_batch_nutrition_detection():
    assert is_batch_nutrition({"items": [{"food_display_name": "a"}, {"food_display_name": "b"}]})
    assert not is_batch_nutrition({"food_display_name": "solo"})


def test_route_nutrition_lookup_only():
    assert route_after_intent(Intent.QUERY_NUTRITION.value, {"food_display_name": "banana"}) == "lookup_nutrition"


def test_route_log_nutrition_with_resolved_macros_executes():
    payload = {
        "food_display_name": "linguine",
        "calories_kcal": 467,
        "nutrition_resolution": "use_search",
    }
    assert route_after_intent(Intent.LOG_NUTRITION.value, payload) == "execute_health"


def test_route_log_nutrition_with_router_shorthand_macros_lookups():
    payload = {"food_display_name": "linguine", "calories_kcal": 467}
    assert route_after_intent(Intent.LOG_NUTRITION.value, payload) == "lookup_nutrition"


def test_route_batch_nutrition():
    payload = {"items": [{"food_display_name": "eggs"}, {"food_display_name": "toast"}]}
    assert route_after_intent(Intent.LOG_NUTRITION.value, payload) == "batch_log_nutrition"


def test_route_research_terminal():
    assert route_after_intent(Intent.GENERAL_RESEARCH.value, {}) == "research_answer"


def test_route_local_intents():
    assert route_after_intent(Intent.LOG_MOOD.value, {}) == "execute_health"
    assert route_after_intent(Intent.UNDO_LAST_LOG.value, {}) == "execute_health"


def test_route_coach_data():
    assert route_after_intent(Intent.QUERY_COACH_DATA.value, {}) == "query_coach_data"


def test_route_wellness_plan():
    assert route_after_intent(Intent.BUILD_WELLNESS_PLAN.value, {}) == "build_wellness_plan"


def test_route_coaching_chat_finalize():
    assert route_after_intent(Intent.COACHING_CHAT.value, {}) == "finalize_reply"


def test_after_nutrition_lookup_skips_sync_for_query():
    payload = {"nutrition_lookup_only": True}
    assert route_after_nutrition_lookup(Intent.QUERY_NUTRITION.value, payload) == "finalize_reply"


def test_capability_confirm_flag():
    cap = get_capability(Intent.LOG_NUTRITION.value)
    assert cap.requires_confirm is True
    assert cap.supports_batch is True
