from backend.health_coach.integrations.nutrition import (
    _prefer_food_relevant_results,
    apply_user_stated_macros,
    build_nutrition_query,
    build_nutrition_user_reply,
    compose_nutrition_reply,
    extract_user_stated_macros,
    format_tavily_source_links,
    needs_nutrition_lookup,
    search_food_nutrition,
    search_has_usable_results,
    should_skip_health_sync,
)


def test_build_nutrition_query_includes_food_and_portion():
    query = build_nutrition_query(
        food_display_name="chapati",
        portion_description="2 whole wheat chapatis",
        user_message="had 2 chapatis for dinner",
    )
    assert "chapati" in query
    assert "2 whole wheat chapatis" in query
    assert "nutrition facts" in query


def test_build_nutrition_query_compacts_grams_each_portion():
    query = build_nutrition_query(
        food_display_name="apple",
        portion_description="2 medium apples (about 182g each)",
        user_message="what about 2 apples?",
    )
    assert query.startswith("364 grams apple")
    assert "what about" not in query
    assert "(" not in query


def test_needs_nutrition_lookup_for_log_and_time_only_update():
    assert needs_nutrition_lookup(
        "LOG_NUTRITION",
        {"food_display_name": "oats", "meal_type": "BREAKFAST"},
    )
    assert needs_nutrition_lookup(
        "QUERY_NUTRITION",
        {"food_display_name": "banana", "portion_description": "1 medium"},
    )
    assert not needs_nutrition_lookup(
        "UPDATE_NUTRITION",
        {"logged_at_hkt": "2026-06-08T22:30:00"},
    )
    assert needs_nutrition_lookup(
        "UPDATE_NUTRITION",
        {"food_display_name": "chapati", "portion_description": "2 chapatis"},
    )


def test_search_food_nutrition_without_api_key(monkeypatch):
    from backend.health_coach.core import database

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = search_food_nutrition(
        food_display_name="banana",
        portion_description="1 medium banana",
    )
    assert result["status"] == "missing_api_key"
    assert "banana" in result["query"]
    rows = database.fetch_recent("tavily_calls", limit=5)
    assert rows[0]["status"] == "missing_api_key"
    assert "banana" in rows[0]["query"]


def test_search_food_nutrition_success(monkeypatch):
    import sys
    import types

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("ENABLE_NUTRITION_SEARCH", "true")

    class FakeTavilyClient:
        def __init__(self, api_key: str):
            self.api_key = api_key

        def search(self, query, **kwargs):
            assert "banana" in query
            assert kwargs["include_domains"]
            return {
                "query": query,
                "answer": "A medium banana has about 105 kcal.",
                "results": [
                    {
                        "title": "Banana nutrition",
                        "url": "https://fdc.nal.usda.gov/banana",
                        "content": "105 kcal, 1.3g protein, 27g carbs, 0.4g fat",
                        "score": 0.92,
                    }
                ],
                "response_time": 0.4,
            }

    fake_module = types.ModuleType("tavily")
    fake_module.TavilyClient = FakeTavilyClient
    monkeypatch.setitem(sys.modules, "tavily", fake_module)

    result = search_food_nutrition(
        food_display_name="banana",
        portion_description="1 medium banana",
    )
    assert result["status"] == "success"
    assert result["answer"]
    assert len(result["results"]) == 1


def test_search_has_usable_results():
    assert search_has_usable_results({"status": "success", "answer": "105 kcal", "results": []})
    assert search_has_usable_results(
        {"status": "success", "results": [{"content": "macros", "url": "https://usda.gov"}]}
    )
    assert not search_has_usable_results({"status": "error", "results": []})
    assert not search_has_usable_results({"status": "success", "results": [{}]})


def test_prefer_food_relevant_results_filters_mixed_pages():
    response = {
        "results": [
            {"title": "Calories in Apples", "url": "https://example.com/apples", "content": "apple macros"},
            {"title": "Ground beef nutrition", "url": "https://example.com/beef", "content": "beef macros"},
        ]
    }
    filtered = _prefer_food_relevant_results(response, "apple")
    assert len(filtered["results"]) == 1
    assert "Apple" in filtered["results"][0]["title"] or "apple" in filtered["results"][0]["content"]


def test_should_skip_health_sync_for_query_and_followup():
    assert should_skip_health_sync("QUERY_NUTRITION", {"nutrition_resolution": "use_search"})
    assert should_skip_health_sync("LOG_NUTRITION", {"nutrition_resolution": "ask_followup"})
    assert should_skip_health_sync("LOG_NUTRITION", {"nutrition_resolution": "use_search"})
    assert not should_skip_health_sync(
        "LOG_NUTRITION",
        {"nutrition_resolution": "use_search", "calories_kcal": 650},
    )


def test_format_tavily_source_links():
    text = format_tavily_source_links(
        {
            "results": [
                {"title": "USDA Banana", "url": "https://fdc.nal.usda.gov/banana"},
                {"title": "No URL"},
            ]
        }
    )
    assert "https://fdc.nal.usda.gov/banana" in text
    assert "USDA Banana" in text


def test_build_nutrition_user_reply_use_search_includes_url():
    reply = build_nutrition_user_reply(
        {
            "nutrition_resolution": "use_search",
            "calories_kcal": 240,
            "nutrition_source": "USDA FoodData Central",
            "nutrition_source_url": "https://fdc.nal.usda.gov/food/123",
            "nutrition_sanity_check": "Looks reasonable for 2 chapatis.",
        }
    )
    assert "240" in reply
    assert "USDA" in reply
    assert "https://fdc.nal.usda.gov/food/123" in reply
    assert "reasonable" in reply


def test_build_nutrition_user_reply_educated_guess():
    reply = build_nutrition_user_reply(
        {
            "nutrition_resolution": "educated_guess",
            "calories_kcal": 500,
            "nutrition_notes": "Assumed a standard restaurant portion.",
        }
    )
    assert "educated estimate" in reply
    assert "500" in reply


def test_build_nutrition_user_reply_missing_calories_lookup_only():
    reply = build_nutrition_user_reply(
        {
            "nutrition_resolution": "educated_guess",
            "nutrition_lookup_only": True,
        }
    )
    assert "log it" in reply.lower()
    assert "Logged" not in reply
    assert "None" not in reply


def test_should_skip_health_sync_without_calories():
    assert should_skip_health_sync(
        "LOG_NUTRITION",
        {"nutrition_resolution": "educated_guess", "calories_kcal": None},
    )


def test_nutrition_macros_response_accepts_null_source_url():
    from backend.health_coach.agent.engine import NutritionMacrosResponse

    parsed = NutritionMacrosResponse.model_validate(
        {
            "resolution": "educated_guess",
            "calories_kcal": 890,
            "protein_grams": 33,
            "carbs_grams": 70,
            "fat_grams": 45,
            "food_display_name": "Lamb curry meal",
            "nutrition_source": "",
            "source_url": None,
            "source_urls": [],
            "nutrition_reply": "Logged ~890 kcal estimate.",
        }
    )
    assert parsed.source_url == ""
    assert parsed.calories_kcal == 890


def test_build_nutrition_user_reply_lookup_only_includes_url():
    reply = build_nutrition_user_reply(
        {
            "nutrition_resolution": "use_search",
            "nutrition_lookup_only": True,
            "calories_kcal": 105,
            "nutrition_source": "USDA",
            "nutrition_source_url": "https://fdc.nal.usda.gov/banana",
        }
    )
    assert "About" in reply
    assert "Logged" not in reply
    assert "https://fdc.nal.usda.gov/banana" in reply
    assert "log it" in reply.lower()


def test_compose_nutrition_reply_prefers_llm_message():
    reply = compose_nutrition_reply(
        base_reply="Got it — logging your dinner.",
        resolved={
            "nutrition_reply": "Logged ~650 kcal from Nutritionix (https://nutritionix.com/food/1).",
            "nutrition_resolution": "use_search",
        },
    )
    assert reply.startswith("Got it")
    assert "https://nutritionix.com/food/1" in reply


def test_extract_user_stated_macros_from_message():
    stated = extract_user_stated_macros(
        "having a coco cream dream protein shake that has a 48g protein w 650 calories now",
        "",
    )
    assert stated["calories_kcal"] == 650
    assert stated["protein_grams"] == 48


def test_extract_user_stated_macros_from_portion_description():
    stated = extract_user_stated_macros("", "48g protein, 650 calories")
    assert stated["calories_kcal"] == 650
    assert stated["protein_grams"] == 48


def test_apply_user_stated_macros_overrides_ask_followup():
    resolved = apply_user_stated_macros(
        {
            "food_display_name": "Coco Cream Dream Protein Shake",
            "portion_description": "48g protein, 650 calories",
            "nutrition_resolution": "ask_followup",
            "nutrition_reply": "Could you clarify the scoop count?",
        },
        user_text="having a shake with 48g protein and 650 calories",
    )
    assert resolved["nutrition_resolution"] == "user_stated"
    assert resolved["calories_kcal"] == 650
    assert resolved["protein_grams"] == 48
    assert should_skip_health_sync("LOG_NUTRITION", resolved) is False


def test_apply_user_stated_macros_skips_meal_total_in_batch_item_context():
    resolved = apply_user_stated_macros(
        {
            "food_display_name": "satay beef noodle soup",
            "nutrition_resolution": "use_search",
            "calories_kcal": 350,
        },
        user_text="it should be around 700 calories for the whole breakfast, search individually",
        item_context=True,
    )
    assert resolved["calories_kcal"] == 350
    assert resolved["nutrition_resolution"] == "use_search"


def test_compose_nutrition_reply_drops_false_log_preamble_on_followup():
    reply = compose_nutrition_reply(
        base_reply="Got it! I'm logging your shake to your health app.",
        resolved={
            "nutrition_resolution": "ask_followup",
            "nutrition_reply": "How many scoops did you use?",
        },
    )
    assert "logging your shake" not in reply
    assert "How many scoops" in reply


def test_build_nutrition_user_reply_user_stated():
    reply = build_nutrition_user_reply(
        {
            "nutrition_resolution": "user_stated",
            "calories_kcal": 650,
            "protein_grams": 48,
            "carbs_grams": 47,
            "fat_grams": 30,
        }
    )
    assert "650" in reply
    assert "48g protein" in reply
    assert "numbers you provided" in reply
