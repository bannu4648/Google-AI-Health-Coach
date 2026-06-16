from backend.health_coach.services.fitness_plans import (
    format_full_plan_for_reply,
    get_active_plan,
    get_relevant_active_plan,
    save_fitness_plan,
)
from backend.health_coach.services.wellness_logs import (
    fetch_recent_moods,
    log_mood,
    summarize_mood_trend,
)


def test_save_and_fetch_fitness_plan():
    saved = save_fitness_plan(
        week_start_hkt="2026-06-09",
        goals={"focus": "strength"},
        weekly_targets={"workouts": 4},
        workouts=[
            {
                "day_of_week": 0,
                "title": "Upper body",
                "exercise_type": "STRENGTH_TRAINING",
                "duration_minutes": 40,
                "steps": ["Warm up 5 min", "Push-ups 3x12"],
            }
        ],
    )
    assert saved is not None
    plan = get_active_plan(week_start_hkt="2026-06-09")
    assert plan is not None
    assert len(plan["workouts"]) == 1


def test_log_mood_and_summarize():
    log_mood(logged_at_hkt="2026-06-13T18:00:00", mood_level=4, notes="good day")
    entries = fetch_recent_moods(limit=5)
    assert entries
    assert "mood logs" in summarize_mood_trend(entries).lower()


def test_relevant_plan_lookup_upcoming_week():
    save_fitness_plan(
        week_start_hkt="2026-06-23",
        goals={"focus": "cardio"},
        weekly_targets={},
        workouts=[
            {
                "day_of_week": 2,
                "title": "Run day",
                "exercise_type": "RUNNING",
                "duration_minutes": 30,
                "steps": ["Easy jog 20 min"],
            }
        ],
    )
    plan = get_relevant_active_plan()
    assert plan is not None
    formatted = format_full_plan_for_reply(plan)
    assert "Week of" in formatted
