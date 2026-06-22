"""
End-to-end style simulations for the nutrition lookup → resolve → log flow.

Uses mocked LLM / Google Health / Tavily — no live API keys required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from backend.health_coach.agent.actions import _delete_nutrition_logs
from backend.health_coach.agent.engine import AIEngine, Intent, NutritionMacrosResponse
from backend.health_coach.agent.graph import _build_invoke_input, _graph_config, build_coach_graph
from backend.health_coach.agent.vision import VisionAnalysis
from backend.health_coach.integrations.nutrition import (
    compose_nutrition_reply,
    search_has_usable_results,
    should_skip_health_sync,
)
from backend.health_coach.services.pending_actions import (
    clear_pending_nutrition,
    is_log_followup_text,
    load_pending_nutrition,
    save_pending_nutrition,
)


TAVILY_LAMB = {
    "status": "success",
    "query": "lamb curry meal nutrition facts",
    "answer": "A typical lamb curry meal with rice and naan is about 1000-1200 kcal, 30-40g protein.",
    "results": [
        {
            "title": "Lamb Curry with Rice",
            "url": "https://www.arise-app.com/dish/Lamb-Curry-with-Rice",
            "content": "870 calories per serving, 39g protein, 78g carbs, 44g fat.",
        },
        {
            "title": "Lamb Curry",
            "url": "https://foods.fatsecret.com/calories-nutrition/generic/lamb-curry",
            "content": "257 calories in 1 cup of Lamb Curry, 28g protein.",
        },
    ],
}


class FakeLLM:
    """Minimal LLM stub for graph simulations."""

    provider_name = "fake"
    model_name = "fake-model"
    rate_limit_user_reply = "rate limited"

    def __init__(self, *, json_responses: dict[str, Any] | None = None):
        self._json_responses = json_responses or {}
        self.calls: list[str] = []

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        return False

    def generate_json(self, *, purpose: str, system_prompt: str, user_prompt: str, temperature: float = 0.2, images=None):
        self.calls.append(purpose)
        if purpose in self._json_responses:
            value = self._json_responses[purpose]
            if isinstance(value, Exception):
                raise value
            return value
        return {
            "intent": "LOG_NUTRITION",
            "payload": {"food_display_name": "meal"},
            "conversational_reply": "Logging your meal.",
        }

    def generate_structured(self, *, purpose: str, system_prompt: str, user_prompt: str, response_model, temperature: float = 0.2, images=None):
        raw = self.generate_json(
            purpose=purpose,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            images=images,
        )
        try:
            return response_model.model_validate(raw)
        except (ValidationError, ValueError):
            return None


@pytest.mark.parametrize(
    "raw,expected_kcal",
    [
        (
            {
                "resolution": "educated_guess",
                "calories_kcal": 890,
                "protein_grams": 33,
                "food_display_name": "Lamb curry",
                "source_url": None,
                "source_urls": None,
                "nutrition_reply": "Logged ~890 kcal estimate.",
            },
            890,
        ),
        (
            {
                "resolution": "use_search",
                "calories_kcal": 1160.7,
                "protein_grams": "47",
                "food_display_name": "Lamb curry",
                "source_url": "https://example.com/lamb",
                "nutrition_reply": "Logged ~1161 kcal.",
            },
            1161,
        ),
        (
            {
                "resolution": "USE_SEARCH",
                "calories_kcal": "870",
                "food_display_name": "Lamb curry with rice",
                "source_url": "https://example.com/lamb",
                "nutrition_reply": "Logged ~870 kcal.",
            },
            870,
        ),
    ],
)
def test_nutrition_macros_response_coerces_gemini_shapes(raw, expected_kcal):
    parsed = NutritionMacrosResponse.model_validate(raw)
    assert parsed.calories_kcal == expected_kcal
    assert parsed.source_url is None or isinstance(parsed.source_url, str)


def test_resolve_nutrition_macros_accepts_educated_guess_with_null_source_url():
    llm = FakeLLM(
        json_responses={
            "resolve_nutrition_macros": {
                "resolution": "educated_guess",
                "calories_kcal": 938,
                "protein_grams": 42,
                "carbs_grams": 85,
                "fat_grams": 48,
                "food_display_name": "Lamb curry meal",
                "nutrition_source": "",
                "source_url": None,
                "source_urls": [],
                "nutrition_reply": "Logged ~938 kcal estimate for your lamb curry meal.",
            }
        }
    )
    engine = AIEngine(llm=llm)
    result = engine.resolve_nutrition_macros(
        user_text="half rice and half naan",
        payload={"food_display_name": "Lamb curry meal", "meal_type": "DINNER"},
        search_result=TAVILY_LAMB,
        intent="LOG_NUTRITION",
    )
    assert result["calories_kcal"] == 938
    assert result["nutrition_resolution"] == "educated_guess"
    assert not should_skip_health_sync("LOG_NUTRITION", result)


def test_resolve_nutrition_macros_empty_llm_json_returns_ask_followup():
    llm = FakeLLM(json_responses={"resolve_nutrition_macros": {}})
    engine = AIEngine(llm=llm)
    result = engine.resolve_nutrition_macros(
        user_text="log dinner",
        payload={"food_display_name": "Lamb curry"},
        search_result=TAVILY_LAMB,
        intent="LOG_NUTRITION",
    )
    assert result["nutrition_resolution"] == "ask_followup"
    assert should_skip_health_sync("LOG_NUTRITION", result)
    reply = compose_nutrition_reply(base_reply="", resolved=result)
    assert "None" not in reply
    assert "logged" not in reply.lower() or "nothing was logged" in reply.lower()


def test_resolve_nutrition_macros_query_mode_lookup_only_on_failure():
    llm = FakeLLM(json_responses={"resolve_nutrition_macros": {}})
    engine = AIEngine(llm=llm)
    result = engine.resolve_nutrition_macros(
        user_text="eating lamb curry for dinner",
        payload={"food_display_name": "Lamb curry"},
        search_result=TAVILY_LAMB,
        intent="QUERY_NUTRITION",
    )
    assert result["nutrition_lookup_only"] is True
    assert should_skip_health_sync("QUERY_NUTRITION", result)


def test_compose_reply_never_shows_none_kcal():
    reply = compose_nutrition_reply(
        base_reply="Nice dinner photo.",
        resolved={
            "nutrition_resolution": "educated_guess",
            "nutrition_lookup_only": True,
            "calories_kcal": None,
        },
    )
    assert "None" not in reply
    assert "log it" in reply.lower()


def test_tavily_lamb_fixture_is_usable():
    assert search_has_usable_results(TAVILY_LAMB)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("log it", True),
        ("add as a new log!! fetch calories", True),
        ("and log it.", True),
        ("the gym is closed", False),
    ],
)
def test_log_followup_phrases(text, expected):
    assert is_log_followup_text(text) is expected


def test_photo_query_flow_saves_pending_and_skips_health_sync(monkeypatch):
    monkeypatch.setattr(
        "backend.health_coach.agent.graph.search_food_nutrition",
        lambda **kwargs: dict(TAVILY_LAMB),
    )
    llm = FakeLLM(
        json_responses={
            "analyze_food_image": {
                "food_display_name": "Lamb curry plate",
                "portion_description": "1 plate",
                "meal_type": "DINNER",
                "wants_to_log": False,
                "lookup_only": True,
                "confidence": "high",
                "vision_notes": "",
                "conversational_reply": "I see lamb curry with rice and naan.",
            },
            "resolve_nutrition_macros": {
                "resolution": "use_search",
                "calories_kcal": 870,
                "protein_grams": 39,
                "carbs_grams": 78,
                "fat_grams": 44,
                "food_display_name": "Lamb curry plate",
                "nutrition_source": "Arise",
                "source_url": "https://www.arise-app.com/dish/Lamb-Curry-with-Rice",
                "source_urls": ["https://www.arise-app.com/dish/Lamb-Curry-with-Rice"],
                "nutrition_reply": "About ~870 kcal (39g protein) — Arise: https://www.arise-app.com/dish/Lamb-Curry-with-Rice Say 'log it' if you want this saved.",
            },
        }
    )
    health = MagicMock()
    health.create_data_point.side_effect = AssertionError("should not write on QUERY")

    graph = build_coach_graph(ai_engine=AIEngine(llm=llm), health_client=health, checkpointer=InMemorySaver())
    phone = "sim_photo_query"
    clear_pending_nutrition(phone)

    result = graph.invoke(
        _build_invoke_input(
            user_text="",
            sender_phone=phone,
            message_type="image",
            image_bytes=b"fake",
            image_caption="eating this lamb curry for dinner",
        ),
        config=_graph_config(phone),
    )

    assert result.get("intent") == "QUERY_NUTRITION"
    assert "870" in (result.get("final_reply") or "")
    assert "log it" in (result.get("final_reply") or "").lower()
    pending = load_pending_nutrition(phone)
    assert pending is not None
    assert pending["payload"]["food_display_name"] == "Lamb curry plate"
    health.create_data_point.assert_not_called()
    clear_pending_nutrition(phone)


def test_log_followup_triggers_confirm_when_low_confidence(monkeypatch):
    """Educated guesses default to low confidence → confirm interrupt before sync."""
    monkeypatch.setattr(
        "backend.health_coach.agent.graph.search_food_nutrition",
        lambda **kwargs: dict(TAVILY_LAMB),
    )
    llm = FakeLLM(
        json_responses={
            "resolve_nutrition_macros": {
                "resolution": "educated_guess",
                "calories_kcal": 900,
                "protein_grams": 35,
                "food_display_name": "Lamb curry plate",
                "source_url": None,
                "confidence": "low",
                "nutrition_reply": "Logged ~900 kcal estimate.",
            },
        }
    )
    health = MagicMock()
    phone = "sim_confirm"
    save_pending_nutrition(phone, payload={"food_display_name": "Lamb curry plate", "meal_type": "DINNER"})
    graph = build_coach_graph(ai_engine=AIEngine(llm=llm), health_client=health, checkpointer=InMemorySaver())
    result = graph.invoke(
        _build_invoke_input(user_text="log it", sender_phone=phone, message_type="text"),
        config=_graph_config(phone),
    )
    assert "__interrupt__" in result or result.get("pending_confirm")
    health.create_data_point.assert_not_called()
    clear_pending_nutrition(phone)


def test_log_followup_after_photo_logs_to_health(monkeypatch):
    monkeypatch.setattr(
        "backend.health_coach.agent.graph.search_food_nutrition",
        lambda **kwargs: dict(TAVILY_LAMB),
    )
    llm = FakeLLM(
        json_responses={
            "resolve_nutrition_macros": {
                "resolution": "educated_guess",
                "calories_kcal": 900,
                "protein_grams": 35,
                "carbs_grams": 80,
                "fat_grams": 40,
                "food_display_name": "Lamb curry plate",
                "nutrition_source": "",
                "source_url": None,
                "source_urls": [],
                "confidence": "high",
                "nutrition_reply": "Logged ~900 kcal estimate for lamb curry.",
            },
        }
    )
    health = MagicMock()
    health.create_data_point.return_value = {
        "done": True,
        "response": {"name": "users/me/dataTypes/nutrition-log/dataPoints/999"},
    }

    phone = "sim_log_followup"
    save_pending_nutrition(
        phone,
        payload={
            "food_display_name": "Lamb curry plate",
            "meal_type": "DINNER",
            "portion_description": "1 plate",
        },
        user_text="photo",
    )
    graph = build_coach_graph(ai_engine=AIEngine(llm=llm), health_client=health, checkpointer=InMemorySaver())

    result = graph.invoke(
        _build_invoke_input(user_text="log it", sender_phone=phone, message_type="text"),
        config=_graph_config(phone),
    )

    assert result.get("intent") == "LOG_NUTRITION"
    assert "900" in (result.get("final_reply") or "")
    health.create_data_point.assert_called_once()
    clear_pending_nutrition(phone)


def test_delete_all_matches_removes_orphan(monkeypatch):
    health = MagicMock()
    health.list_all_data_points.return_value = {
        "dataPoints": [
            {
                "name": "users/x/dataTypes/nutrition-log/dataPoints/ghost",
                "nutritionLog": {
                    "foodDisplayName": "Thai spicy minced pork with rice",
                    "interval": {"startTime": "2026-06-16T17:00:04Z", "startUtcOffset": "28800s"},
                },
            }
        ]
    }
    monkeypatch.setattr(
        "backend.health_coach.agent.actions.default_query_range_utc",
        lambda days=3: ("2026-06-16T16:00:00Z", "2026-06-17T16:00:00Z"),
    )
    result = _delete_nutrition_logs(
        {
            "match_keywords": ["thai", "minced pork"],
            "date_hkt": "2026-06-17",
            "delete_all_matches": True,
        },
        client=health,
    )
    assert result["deleted_count"] == 1
    health.batch_delete_data_points.assert_called_once_with(
        ["users/x/dataTypes/nutrition-log/dataPoints/ghost"]
    )


def test_nutrition_retry_guard_reroutes_try_again_to_log():
    from backend.health_coach.agent.graph import _apply_nutrition_retry_guard
    from backend.health_coach.agent.engine import Intent

    assert _apply_nutrition_retry_guard("try again now", Intent.UPDATE_NUTRITION.value) == Intent.LOG_NUTRITION.value
    assert (
        _apply_nutrition_retry_guard(
            "they are wrongly mapped as 15th june",
            Intent.UPDATE_NUTRITION.value,
        )
        == Intent.UPDATE_NUTRITION.value
    )


def test_vision_structured_model_used_by_agent():
    client = MagicMock()
    client.generate_structured.return_value = VisionAnalysis(
        food_display_name="Lamb curry",
        portion_description="1 plate",
        meal_type="DINNER",
        wants_to_log=False,
        lookup_only=True,
        conversational_reply="Lamb curry plate — I'll look up nutrition.",
    )
    from backend.health_coach.agent.vision import VisionAgent

    agent = VisionAgent(client=client)
    result = agent.analyze_food_image(image_bytes=b"x", mime_type="image/jpeg", caption="dinner")
    assert result["lookup_only"] is True
    assert result["food_display_name"] == "Lamb curry"
