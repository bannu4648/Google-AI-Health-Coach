from backend.health_coach.core.health_normalizer import normalize_health_result


def test_exercise_normalizer_includes_notes_and_display_name():
    result = normalize_health_result(
        "exercise",
        {
            "dataPoints": [
                {
                    "exercise": {
                        "displayName": "Home Bodyweight Workout",
                        "exerciseType": "STRENGTH_TRAINING",
                        "activeDuration": "1500s",
                        "notes": "3x12 squats, 3x12 push-ups",
                        "interval": {
                            "startTime": "2026-06-16T15:34:00Z",
                            "endTime": "2026-06-16T15:59:00Z",
                        },
                        "metricsSummary": {"caloriesKcal": 146},
                    },
                    "dataSource": {"platform": "GOOGLE_WEB_API"},
                }
            ]
        },
    )
    record = result["records"][0]
    assert record["display_name"] == "Home Bodyweight Workout"
    assert record["notes"] == "3x12 squats, 3x12 push-ups"
    assert record["active_duration_minutes"] == 25
    assert record["calories_kcal"] == 146
