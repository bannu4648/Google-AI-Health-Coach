from backend.health_coach.services import coaching


def test_daily_health_snapshot_cache(monkeypatch):
    calls = {"count": 0}

    def fake_fetch(health, *, start, end, include_rollups=True):
        calls["count"] += 1
        return {"range_utc": {"start": start, "end": end}, "steps": {"count": 1}}

    monkeypatch.setattr(coaching, "_fetch_range_metrics", fake_fetch)
    monkeypatch.setattr(coaching, "HEALTH_SNAPSHOT_CACHE_SECONDS", 60)
    coaching.clear_health_snapshot_cache()

    first = coaching.get_daily_health_snapshot()
    second = coaching.get_daily_health_snapshot()

    assert calls["count"] == 1
    assert first["date_hkt"] == second["date_hkt"]
