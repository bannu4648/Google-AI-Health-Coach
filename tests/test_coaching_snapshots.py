from datetime import datetime, time, timedelta
from unittest.mock import MagicMock

from backend.health_coach.core.timezone import get_user_tz
from backend.health_coach.services import coaching


def _empty_roll_up():
    return {"rollupDataPoints": []}


def _empty_list():
    return {"dataPoints": []}


def _mock_health_client():
    client = MagicMock()
    client.daily_roll_up.return_value = _empty_roll_up()
    client.list_data_points.return_value = _empty_list()
    client.reconcile_data_points.return_value = _empty_list()
    return client


def test_evening_snapshot_is_today_only(monkeypatch):
    client = _mock_health_client()
    snapshot = coaching.get_evening_health_snapshot(client=client)

    assert snapshot["summary_type"] == "evening"
    assert snapshot["scope"] == "today_only"
    assert "steps" in snapshot
    assert "weekly_trends" not in snapshot
    client.daily_roll_up.assert_called()
    client.list_data_points.assert_called()


def test_morning_snapshot_includes_sleep_window_and_weekly_trends(monkeypatch):
    client = _mock_health_client()
    snapshot = coaching.get_morning_health_snapshot(client=client)

    assert snapshot["summary_type"] == "morning"
    assert snapshot["scope"] == "last_night_sleep_and_weekly_trends"
    assert "last_night_sleep" in snapshot
    assert snapshot["weekly_trends"]["days"] == coaching.WEEKLY_LOOKBACK_DAYS
    assert "today_so_far" in snapshot

    sleep_calls = [
        call.kwargs
        for call in client.reconcile_data_points.call_args_list
        if call.args and call.args[0] == "sleep"
    ]
    assert sleep_calls, "expected at least one sleep reconcile call"


def test_last_night_sleep_bounds_cover_previous_evening(monkeypatch):
    fixed_now = datetime(2026, 6, 11, 8, 0, tzinfo=get_user_tz())
    monkeypatch.setattr(coaching, "now_local", lambda: fixed_now)

    start, end = coaching.last_night_sleep_bounds_utc()
    assert "2026-06-10" in start
    assert "18:00:00" in start or "10:00:00" in start  # UTC offset for HKT
    assert "2026-06-11" in end


def test_readiness_uses_sleep_hours():
    snapshot = {
        "steps": {"count": 5000},
        "exercise": {"count": 0},
        "sleep": {"count": 1, "duration_hours": 5.2, "deep_minutes": 25},
        "active_zone_minutes": {"total": 10},
        "resting_heart_rate": {"bpm": 72},
        "nutrition": {"count": 2},
    }
    result = coaching.readiness_score(snapshot)
    assert result["label"] in {"steady", "recover", "ready"}
    assert any("5.2" in reason for reason in result["reasons"])


def test_morning_readiness_uses_last_night_sleep():
    snapshot = {
        "last_night_sleep": {"count": 1, "duration_hours": 8.1, "rem_minutes": 95},
        "weekly_trends": {
            "steps": {"average_on_active_days": 9000},
            "exercise": {"total_sessions": 4},
            "active_zone_minutes": {"daily": [{"value": 15}]},
        },
    }
    result = coaching.morning_readiness_score(snapshot)
    assert result["score"] >= 55
    assert any("8.1" in reason or "sleep" in reason.lower() for reason in result["reasons"])


def test_create_daily_summary_branches_on_type(monkeypatch):
    client = _mock_health_client()
    monkeypatch.setattr(
        coaching,
        "build_daily_coach_message",
        lambda st, snap, client=None: f"{st}-message",
    )

    evening = coaching.create_daily_summary("evening", client=client)
    morning = coaching.create_daily_summary("morning", client=client)

    assert evening["message"] == "evening-message"
    assert morning["message"] == "morning-message"
    assert evening["snapshot"]["scope"] == "today_only"
    assert morning["snapshot"]["scope"] == "last_night_sleep_and_weekly_trends"
