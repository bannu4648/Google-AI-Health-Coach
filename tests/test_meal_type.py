from backend.health_coach.core.payloads import (
    build_nutrition_data_point,
    expand_nutrition_items,
    normalize_meal_type,
)


def test_normalize_meal_type_unknown_to_unspecified():
    assert normalize_meal_type("UNKNOWN") == "MEAL_TYPE_UNSPECIFIED"
    assert normalize_meal_type(None) == "MEAL_TYPE_UNSPECIFIED"
    assert normalize_meal_type("") == "MEAL_TYPE_UNSPECIFIED"


def test_normalize_meal_type_drinks_to_snack():
    assert normalize_meal_type("DRINK") == "SNACK"
    assert normalize_meal_type("white wine") == "SNACK"


def test_normalize_meal_type_passes_valid_enum():
    assert normalize_meal_type("DINNER") == "DINNER"
    assert normalize_meal_type("snack") == "SNACK"


def test_build_nutrition_data_point_uses_normalized_meal_type():
    point = build_nutrition_data_point(
        {
            "food_display_name": "white wine",
            "meal_type": "UNKNOWN",
            "calories_kcal": 240,
        }
    )
    assert point["nutritionLog"]["mealType"] == "MEAL_TYPE_UNSPECIFIED"


def test_expand_nutrition_items_batch():
    payload = {
        "items": [
            {"food_display_name": "gin and tonic", "portion_description": "2"},
            {"food_display_name": "white wine", "portion_description": "2 glasses"},
        ],
        "logged_at_hkt": "2026-06-12T19:00:00",
    }
    items = expand_nutrition_items(payload)
    assert len(items) == 2
    assert items[0]["food_display_name"] == "gin and tonic"
    assert items[1]["logged_at_hkt"] == "2026-06-12T19:00:00"


def test_expand_nutrition_items_single():
    payload = {"food_display_name": "chapati", "portion_description": "2"}
    assert len(expand_nutrition_items(payload)) == 1
