from backend.health_coach.services.memory import format_history_for_prompt
from backend.health_coach.services.user_profile import format_user_profile_for_prompt


def test_format_history_excludes_current_user_message(monkeypatch):
    monkeypatch.setattr(
        "backend.health_coach.services.memory.fetch_recent_messages_for_phone",
        lambda phone, limit: [
            {"direction": "inbound", "text": "log my lunch"},
            {"direction": "outbound", "text": "Logged lunch."},
            {"direction": "inbound", "text": "what about dinner?"},
        ],
    )
    history = format_history_for_prompt("85200000000", exclude_user_text="what about dinner?")
    assert "what about dinner?" not in history
    assert "log my lunch" in history
    assert "Logged lunch." in history


def test_format_user_profile_for_prompt_includes_known_fields():
    text = format_user_profile_for_prompt(
        {"age_years": 34, "height_cm": 175.0, "weight_kg": 75.5}
    )
    assert "Age: 34" in text
    assert "Height: 175" in text
    assert "Latest weight: 75.5 kg" in text


def test_format_user_profile_for_prompt_handles_missing_data():
    text = format_user_profile_for_prompt({})
    assert "No age/height/weight" in text
