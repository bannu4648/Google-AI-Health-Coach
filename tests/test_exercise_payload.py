from backend.health_coach.agent.engine import Intent, _coerce_router_parsed
from backend.health_coach.core.payloads import build_exercise_data_point, normalize_exercise_type


def test_coerce_router_parsed_list_to_batch():
    parsed = _coerce_router_parsed(
        [
            {"food_display_name": "gin and tonic", "portion_description": "2"},
            {"food_display_name": "white wine", "portion_description": "2 glasses"},
        ]
    )
    assert parsed["intent"] == Intent.LOG_NUTRITION.value
    assert len(parsed["payload"]["items"]) == 2


def test_normalize_exercise_type_aliases():
    assert normalize_exercise_type("run") == "RUNNING"
    assert normalize_exercise_type("gym") == "STRENGTH_TRAINING"


def test_build_exercise_data_point_estimates_calories():
    point = build_exercise_data_point(
        {
            "display_name": "Strength session",
            "exercise_type": "STRENGTH_TRAINING",
            "duration_minutes": 25,
        },
        weight_kg=70.0,
    )
    exercise = point["exercise"]
    assert exercise["metricsSummary"]["caloriesKcal"] == 146


def test_estimate_exercise_calories_respects_explicit_value():
    from backend.health_coach.core.payloads import estimate_exercise_calories_kcal

    assert estimate_exercise_calories_kcal({"calories_kcal": 200, "duration_minutes": 30}) == 200


def test_build_exercise_data_point():
    point = build_exercise_data_point(
        {
            "display_name": "Morning run",
            "exercise_type": "RUNNING",
            "duration_minutes": 30,
            "calories_kcal": 250,
        }
    )
    exercise = point["exercise"]
    assert exercise["displayName"] == "Morning run"
    assert exercise["exerciseType"] == "RUNNING"
    assert exercise["metricsSummary"]["caloriesKcal"] == 250


def test_fix_exercise_data_point_active_energy():
    from backend.health_coach.core.payloads import fix_exercise_data_point_structure

    broken = {
        "exercise": {
            "displayName": "Run",
            "metricsSummary": {"activeEnergy": {"kcal": 120}},
        }
    }
    fixed = fix_exercise_data_point_structure(
        broken,
        'Unknown name "activeEnergy" at metrics_summary',
    )
    assert fixed is not None
    assert fixed["exercise"]["metricsSummary"]["caloriesKcal"] == 120
    assert "activeEnergy" not in fixed["exercise"]["metricsSummary"]


def test_expand_exercise_items_batch():
    from backend.health_coach.core.payloads import expand_exercise_items

    items = expand_exercise_items(
        {
            "items": [
                {"display_name": "Squats", "duration_minutes": 6},
                {"display_name": "Push-ups", "duration_minutes": 6},
            ]
        }
    )
    assert len(items) == 2
    assert items[0]["display_name"] == "Squats"
