"""Tests for nutrition log update/delete matching helpers."""

from backend.health_coach.agent.actions import (
    _pick_best_nutrition_match,
    _update_nutrition_logs,
)


def _point(food: str, start_time: str) -> dict:
    return {
        "name": f"users/x/dataTypes/nutrition-log/dataPoints/{food}",
        "nutritionLog": {
            "foodDisplayName": food,
            "interval": {"startTime": start_time, "startUtcOffset": "28800s"},
        },
    }


def test_pick_best_nutrition_match_prefers_wrong_calendar_day():
    # June 15 15:30 HKT = 2026-06-15T07:30:00Z; target is June 14 same clock time.
    on_june_15 = _point("chicken spaghetti", "2026-06-15T07:30:00Z")
    on_june_14 = _point("chicken spaghetti", "2026-06-14T07:30:00Z")
    picked = _pick_best_nutrition_match(
        [on_june_14, on_june_15],
        target_logged_at_hkt="2026-06-14T15:30:00",
    )
    assert picked is on_june_15


def test_update_nutrition_logs_expands_items(monkeypatch):
    calls: list[str] = []

    def fake_update(payload, *, client, user_text="", exclude_names=None):
        calls.append(payload["food_display_name"])
        return {"message": f"ok:{payload['food_display_name']}"}

    monkeypatch.setattr(
        "backend.health_coach.agent.actions._update_nutrition_log",
        fake_update,
    )
    result = _update_nutrition_logs(
        {
            "items": [
                {"food_display_name": "chicken spaghetti", "logged_at_hkt": "2026-06-14T15:30:00"},
                {"food_display_name": "chicken biryani", "logged_at_hkt": "2026-06-14T21:30:00"},
            ]
        },
        client=object(),
    )
    assert calls == ["chicken spaghetti", "chicken biryani"]
    assert result["updated_count"] == 2
    assert "Updated 2 meal(s)" in result["message"]
