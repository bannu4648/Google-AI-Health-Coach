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
    assert exercise["metricsSummary"]["activeEnergy"]["kcal"] == 250
