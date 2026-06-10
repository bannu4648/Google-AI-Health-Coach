from backend.health_coach.core import database
from backend.health_coach.core.types import normalize_data_type, normalize_query_payload
from backend.health_coach.services import scheduler


def test_legacy_data_types_normalize_to_google_health_v4():
    assert normalize_data_type("ACTIVITY") == "exercise"
    assert normalize_data_type("com.google.step_count.delta") == "steps"
    assert normalize_data_type("resting_heart_rate") == "daily-resting-heart-rate"


def test_exercise_query_trends_reconcile():
    payload = normalize_query_payload({"data_type": "workouts"}, intent="QUERY_TRENDS")
    assert payload["data_type"] == "exercise"
    assert payload["query_method"] == "reconcile"


def test_database_records_event_with_redaction():
    row_id = database.record_event(
        "test_event",
        "pytest",
        status="ok",
        payload={"access_token": "secret-token", "safe": "value"},
    )
    rows = database.fetch_recent("events", limit=5)
    row = next(item for item in rows if item["id"] == row_id)
    assert "secret-token" not in row["payload_json"]
    assert "[REDACTED]" in row["payload_json"]


def test_scheduler_can_create_jobs_when_enabled(monkeypatch):
    monkeypatch.setattr(scheduler, "ENABLE_SCHEDULER", True)
    monkeypatch.setattr(scheduler, "SUMMARY_RECIPIENT_PHONE", "85200000000")
    monkeypatch.setattr(scheduler, "MORNING_SUMMARY_TIME", "08:00")
    monkeypatch.setattr(scheduler, "EVENING_SUMMARY_TIME", "21:30")
    sched = scheduler.start_scheduler()
    assert sched is not None
    job_ids = {job.id for job in sched.get_jobs()}
    assert {"morning_summary", "evening_summary"} <= job_ids
    scheduler.stop_scheduler()
