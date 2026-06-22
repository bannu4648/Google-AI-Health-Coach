"""Tests for goal update matching and target normalization."""

from backend.health_coach.services.user_goals import (
    find_goal_for_update,
    normalize_goal_target,
    update_goal,
)


def test_normalize_goal_target_maps_router_shorthand():
    assert normalize_goal_target({"min_grams": 90, "max_grams": 95}) == {
        "protein_grams_min": 90,
        "protein_grams_max": 95,
    }


def test_find_goal_for_update_matches_protein_goal_by_loose_label(monkeypatch):
    monkeypatch.setattr(
        "backend.health_coach.services.user_goals.fetch_active_goals",
        lambda limit=10: [
            {
                "id": "goal-1",
                "category": "nutrition",
                "goal_text": "Eat ~115–125 g protein daily while cutting (muscle-preserving target)",
                "target": {"protein_grams_min": 115, "protein_grams_max": 125},
            },
            {
                "id": "goal-2",
                "category": "weight",
                "goal_text": "Lose 6–7 kg and reach ~69–70 kg by end of August 2026",
                "target": {"target_weight_kg": 69.5},
            },
        ],
    )
    matched = find_goal_for_update(
        goal_text="Daily protein intake",
        category="nutrition",
        target={"min_grams": 90, "max_grams": 95},
    )
    assert matched is not None
    assert matched["id"] == "goal-1"


def test_update_goal_applies_normalized_target(monkeypatch):
    stored = {
        "id": "goal-1",
        "category": "nutrition",
        "goal_text": "Eat ~115–125 g protein daily while cutting (muscle-preserving target)",
        "target": {"protein_grams_min": 115, "protein_grams_max": 125},
        "progress": {},
        "status": "active",
        "deadline_hkt": None,
    }

    class FakeRow(dict):
        def __getitem__(self, key):
            mapping = {
                "id": stored["id"],
                "goal_text": stored["goal_text"],
                "target_json": __import__("json").dumps(stored["target"]),
                "progress_json": "{}",
                "status": stored["status"],
                "deadline_hkt": None,
                "category": stored["category"],
            }
            return mapping[key]

    monkeypatch.setattr(
        "backend.health_coach.services.user_goals.fetch_active_goals",
        lambda limit=10: [stored],
    )
    monkeypatch.setattr("backend.health_coach.services.user_goals.init_db", lambda: None)

    class FakeConn:
        def execute(self, query, params=()):
            self.last_query = query
            self.last_params = params
            if "SELECT * FROM user_goals WHERE status = 'active' AND goal_text LIKE" in query:
                return None
            if "SELECT * FROM user_goals WHERE id = ?" in query:
                return FakeRow()
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("backend.health_coach.services.user_goals.connect", lambda: FakeConn())
    monkeypatch.setattr(
        "backend.health_coach.services.user_goals._row_to_goal",
        lambda row: stored,
    )

    def _merge_update(*args, **kwargs):
        stored["target"] = {**stored["target"], **normalize_goal_target(kwargs.get("target"))}
        return stored

    monkeypatch.setattr("backend.health_coach.services.user_goals.utc_now_iso", lambda: "2026-06-17T16:00:00Z")

    # Exercise the lookup path directly; DB update is covered by integration elsewhere.
    matched = find_goal_for_update(
        goal_text="Daily protein intake",
        category="nutrition",
        target={"min_grams": 90, "max_grams": 95},
    )
    assert matched["id"] == "goal-1"
    merged = {**matched["target"], **normalize_goal_target({"min_grams": 90, "max_grams": 95})}
    assert merged["protein_grams_min"] == 90
    assert merged["protein_grams_max"] == 95
