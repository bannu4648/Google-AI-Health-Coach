from backend.health_coach.services.nutrition_plan import format_brief_progress_line, sum_today_nutrition


def test_sum_today_nutrition():
    snapshot = {
        "nutrition": {
            "items": [
                {
                    "nutritionLog": {
                        "energy": {"kcal": 450},
                        "nutrients": [{"nutrient": "PROTEIN", "quantity": {"grams": 35}}],
                    }
                },
                {
                    "nutritionLog": {
                        "energy": {"kcal": 520},
                        "nutrients": [{"nutrient": "PROTEIN", "quantity": {"grams": 28}}],
                    }
                },
            ]
        }
    }
    totals = sum_today_nutrition(snapshot)
    assert totals["calories_kcal"] == 970
    assert totals["protein_grams"] == 63
    assert totals["meals_logged"] == 2


def test_format_brief_progress_line():
    snapshot = {
        "date_hkt": "2026-06-16",
        "nutrition": {
            "items": [
                {
                    "nutritionLog": {
                        "energy": {"kcal": 600},
                        "nutrients": [{"nutrient": "PROTEIN", "quantity": {"grams": 40}}],
                    }
                }
            ]
        }
    }
    line = format_brief_progress_line(
        snapshot,
        plan={
            "daily_calories_target": 1900,
            "protein_grams_min": 115,
            "protein_grams_max": 125,
        },
    )
    assert "600/1900" in line
    assert "40/115" in line
