from backend.health_coach.core.database import record_weight_log
from backend.health_coach.services.weight_tracking import should_send_weekly_weight_nudge


def test_should_send_weight_nudge_when_no_local_log(monkeypatch):
    monkeypatch.setattr(
        "backend.health_coach.services.weight_tracking.days_since_last_weight_log",
        lambda **kwargs: None,
    )
    assert should_send_weekly_weight_nudge() is True


def test_should_not_send_weight_nudge_when_recent(monkeypatch):
    monkeypatch.setattr(
        "backend.health_coach.services.weight_tracking.days_since_last_weight_log",
        lambda **kwargs: 2,
    )
    assert should_send_weekly_weight_nudge() is False


def test_record_weight_log_round_trip():
    row_id = record_weight_log(weight_kg=76.0, logged_at_hkt="2026-06-16T09:00:00", source="test")
    assert row_id
