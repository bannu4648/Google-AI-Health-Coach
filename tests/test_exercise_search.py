from backend.health_coach.integrations.exercise import (
    build_exercise_calorie_query,
    needs_exercise_calorie_lookup,
)


def test_build_exercise_calorie_query_includes_weight():
    query = build_exercise_calorie_query(
        display_name="Bodyweight squats",
        exercise_type="STRENGTH_TRAINING",
        duration_minutes=25,
        notes="3x12",
        weight_kg=76,
    )
    assert "76" in query
    assert "calories burned" in query.lower()


def test_needs_exercise_calorie_lookup_when_missing_kcal():
    assert needs_exercise_calorie_lookup(
        "LOG_EXERCISE",
        {"display_name": "Run", "duration_minutes": 30},
    )
    assert not needs_exercise_calorie_lookup(
        "LOG_EXERCISE",
        {"display_name": "Run", "duration_minutes": 30, "calories_kcal": 200},
    )
