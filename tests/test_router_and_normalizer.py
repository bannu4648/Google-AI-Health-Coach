from backend.health_coach.agent.engine import Intent, RouterResponse
from backend.health_coach.agent.graph import _apply_no_log_guard
from backend.health_coach.core.health_normalizer import normalize_health_result


def test_router_response_coerces_nested_conversational_reply():
    routed = RouterResponse.model_validate(
        {
            "intent": "COACHING_CHAT",
            "payload": {},
            "conversational_reply": {"message": "REM sleep is part of normal sleep architecture."},
        }
    )
    assert routed.conversational_reply == "REM sleep is part of normal sleep architecture."


def test_no_log_guard_converts_accidental_nutrition_log_to_lookup():
    intent, payload = _apply_no_log_guard(
        "don't log anything, how many calories in 2 apples?",
        Intent.LOG_NUTRITION.value,
        {"food_display_name": "apple", "portion_description": "2 apples"},
    )
    assert intent == Intent.QUERY_NUTRITION.value
    assert payload["food_display_name"] == "apple"


def test_normalize_sleep_keeps_all_nights_and_stage_totals():
    result = {
        "dataPoints": [
            {
                "sleep": {
                    "interval": {
                        "startTime": "2026-06-08T16:00:00Z",
                        "endTime": "2026-06-09T00:00:00Z",
                    },
                    "type": "STAGES",
                    "stages": [
                        {
                            "startTime": "2026-06-08T16:00:00Z",
                            "endTime": "2026-06-08T17:00:00Z",
                            "type": "REM",
                        }
                    ],
                }
            },
            {
                "sleep": {
                    "interval": {
                        "startTime": "2026-06-09T16:00:00Z",
                        "endTime": "2026-06-10T00:00:00Z",
                    },
                    "type": "STAGES",
                    "stages": [],
                }
            },
        ]
    }
    normalized = normalize_health_result("sleep", result)
    assert normalized["record_count"] == 2
    assert len(normalized["records"]) == 2
    assert normalized["records"][0]["stage_minutes"]["rem"] == 60
    assert normalized["totals"]["duration_minutes"] == 960


def test_normalize_rollup_steps_totals_every_bucket():
    normalized = normalize_health_result(
        "steps",
        {
            "rollupDataPoints": [
                {"civilStartTime": {"date": {"year": 2026, "month": 6, "day": 8}}, "steps": {"countSum": "1200"}},
                {"civilStartTime": {"date": {"year": 2026, "month": 6, "day": 9}}, "steps": {"countSum": "3400"}},
            ]
        },
    )
    assert normalized["record_count"] == 2
    assert normalized["totals"]["total"] == 4600


def test_normalize_exercise_and_nutrition_records():
    exercise = normalize_health_result(
        "exercise",
        {
            "dataPoints": [
                {
                    "exercise": {
                        "interval": {"startTime": "2026-06-09T10:00:00Z", "endTime": "2026-06-09T11:00:00Z"},
                        "exerciseType": "PICKLEBALL",
                        "metricsSummary": {"caloriesKcal": 300, "steps": "2500"},
                    }
                }
            ]
        },
    )
    assert exercise["records"][0]["exercise_type"] == "PICKLEBALL"
    assert exercise["records"][0]["duration_minutes"] == 60

    nutrition = normalize_health_result(
        "nutrition-log",
        {
            "dataPoints": [
                {
                    "nutritionLog": {
                        "foodDisplayName": "Apple",
                        "mealType": "SNACK",
                        "energy": {"kcal": 189},
                        "totalCarbohydrate": {"grams": 50},
                        "totalFat": {"grams": 0.6},
                        "nutrients": [{"nutrient": "PROTEIN", "quantity": {"grams": 0.9}}],
                    }
                }
            ]
        },
    )
    assert nutrition["record_count"] == 1
    assert nutrition["totals"]["calories_kcal"] == 189
