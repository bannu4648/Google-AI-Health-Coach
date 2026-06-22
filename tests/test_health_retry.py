from unittest.mock import MagicMock, patch

import pytest

from backend.health_coach.agent.actions import _create_data_point_with_retry
from backend.health_coach.agent.engine import Intent
from backend.health_coach.core.health_retry import (
    apply_deterministic_payload_fixes,
    is_retryable_health_api_error,
)
from backend.health_coach.integrations.google_health import GoogleHealthAPIError


def test_is_retryable_health_api_error_meal_type():
    assert is_retryable_health_api_error(
        400,
        'Invalid value at \'data_point.nutrition_log.meal_type\' ... "UNKNOWN"',
    )


def test_is_retryable_health_api_error_not_500():
    assert not is_retryable_health_api_error(500, "internal error")


def test_deterministic_fix_meal_type_unknown():
    payload = {"food_display_name": "white wine", "meal_type": "UNKNOWN", "calories_kcal": 200}
    fixed = apply_deterministic_payload_fixes(
        "LOG_NUTRITION",
        payload,
        'Invalid value at meal_type ... "UNKNOWN"',
    )
    assert fixed is not None
    assert fixed["meal_type"] == "MEAL_TYPE_UNSPECIFIED"


def test_create_data_point_with_retry_succeeds_after_deterministic_fix():
    client = MagicMock()
    client.create_data_point.side_effect = [
        GoogleHealthAPIError(400, 'Invalid value at meal_type ... "UNKNOWN"'),
        {"name": "users/me/dataTypes/nutrition-log/dataPoints/abc"},
    ]
    payload = {
        "food_display_name": "white wine",
        "meal_type": "UNKNOWN",
        "calories_kcal": 200,
        "protein_grams": 0,
        "carbs_grams": 5,
        "fat_grams": 0,
    }
    result = _create_data_point_with_retry(Intent.LOG_NUTRITION, payload, client=client)
    assert result["name"].endswith("/abc")
    assert result["_health_sync_retry"]["applied"] is True
    assert client.create_data_point.call_count == 2


def test_create_data_point_with_retry_raises_when_not_retryable():
    client = MagicMock()
    client.create_data_point.side_effect = GoogleHealthAPIError(401, "unauthorized")
    with pytest.raises(GoogleHealthAPIError):
        _create_data_point_with_retry(
            Intent.LOG_NUTRITION,
            {"food_display_name": "x", "meal_type": "DINNER"},
            client=client,
        )


def test_create_exercise_retry_fixes_active_energy_field():
    client = MagicMock()
    client.create_data_point.side_effect = [
        GoogleHealthAPIError(
            400,
            'Unknown name "activeEnergy" at \'data_point.exercise.metrics_summary\'',
        ),
        {"name": "users/me/dataTypes/exercise/dataPoints/xyz"},
    ]
    payload = {
        "display_name": "Bodyweight Squats",
        "exercise_type": "STRENGTH_TRAINING",
        "duration_minutes": 6,
        "data_point": {
            "exercise": {
                "displayName": "Bodyweight Squats",
                "metricsSummary": {"activeEnergy": {"kcal": 50}},
            }
        },
    }
    result = _create_data_point_with_retry(Intent.LOG_EXERCISE, payload, client=client)
    assert "xyz" in result["name"]
    assert client.create_data_point.call_count == 2
    second_call_body = client.create_data_point.call_args_list[1].args[1]
    assert second_call_body["exercise"]["metricsSummary"]["caloriesKcal"] == 50
