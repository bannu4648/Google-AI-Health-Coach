"""Tests for caption portion hint merging."""

from backend.health_coach.services.portion_hints import apply_caption_portion_hints


def test_apply_caption_portion_hints_chicken_only():
    payload = {
        "food_display_name": "Chicken and egg salad with vegetables and peanut dressing",
        "portion_description": "1 bowl (~350g)",
        "vision_notes": "Peanut dressing visible.",
    }
    merged = apply_caption_portion_hints(
        payload,
        "having this 40g chicken and egg salad for lunch!",
    )
    assert merged["protein_portion_grams"] == 40
    assert merged["protein_component"] == "chicken"
    assert "40g chicken" in merged["portion_description"]
    assert "NOT 40g" in merged["portion_description"]
    assert "350g" in merged["portion_description"]


def test_apply_caption_portion_hints_no_match():
    payload = {"food_display_name": "Oats", "portion_description": "1 bowl"}
    assert apply_caption_portion_hints(payload, "breakfast oats") == payload
