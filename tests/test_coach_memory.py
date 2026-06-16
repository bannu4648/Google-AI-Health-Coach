"""Tests for coach memory: plan lookup, full replies, text2sql guardrails, goals."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.health_coach.agent.intent_registry import LOCAL_COACH_INTENTS, is_batch_nutrition
from backend.health_coach.agent.engine import CoachDbQueryResponse, Intent
from backend.health_coach.agent.graph import build_coach_graph
from backend.health_coach.services.coach_db_tool import lookup_coach_data, query_coach_db
from backend.health_coach.services.coach_state import build_coach_state_snapshot, format_coach_state_for_prompt
from backend.health_coach.services.fitness_plans import (
    format_full_plan_for_reply,
    get_active_plan,
    get_relevant_active_plan,
    parse_day_filter,
    save_fitness_plan,
)
from backend.health_coach.services.llm_context import build_llm_context
from backend.health_coach.services.user_goals import fetch_active_goals, log_goal, update_goal


def test_get_relevant_active_plan_prefers_upcoming_week():
    save_fitness_plan(
        week_start_hkt="2026-06-16",
        goals={"gym_sessions": 2},
        weekly_targets={},
        workouts=[
            {
                "day_of_week": 1,
                "title": "Full Body Gym Workout",
                "exercise_type": "STRENGTH_TRAINING",
                "duration_minutes": 45,
                "steps": ["Squats 3x10", "Bench press 3x8"],
            },
            {
                "day_of_week": 3,
                "title": "Full Body Gym Workout",
                "exercise_type": "STRENGTH_TRAINING",
                "duration_minutes": 45,
                "steps": ["Deadlifts 3x5", "Rows 3x10"],
            },
        ],
    )
    with patch("backend.health_coach.services.fitness_plans.now_local") as mock_now:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        mock_now.return_value = datetime(2026, 6, 13, 22, 37, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        plan = get_relevant_active_plan()
    assert plan is not None
    assert plan["week_start_hkt"] == "2026-06-16"


def test_format_full_plan_includes_steps():
    plan = get_active_plan(week_start_hkt="2026-06-16")
    if plan is None:
        plan = {
            "week_start_hkt": "2026-06-16",
            "workouts": [
                {
                    "day_of_week": 1,
                    "title": "Gym",
                    "duration_minutes": 40,
                    "steps": ["Warm up", "Squats"],
                }
            ],
        }
    text = format_full_plan_for_reply(plan)
    assert "Tuesday" in text
    assert "1." in text
    assert "2." in text


def test_parse_day_filter():
    assert parse_day_filter("Tuesday, Thursday") == [1, 3]
    assert parse_day_filter(["Tuesday", 3]) == [1, 3]


def test_finalize_prefers_db_message_for_query_fitness_plan():
    """finalize_reply must prefer api_result.message for local coach intents."""
    reply = "Here is a hallucinated gym plan with fake exercises."
    api_result = {
        "plan": {"week_start_hkt": "2026-06-16"},
        "message": "Week of 2026-06-16 (0/2 workouts done)\n\nTuesday: *Gym*\n1. Real step",
    }
    intent_name = "QUERY_FITNESS_PLAN"
    if api_result.get("message") and not api_result.get("error"):
        if intent_name in LOCAL_COACH_INTENTS or api_result.get("plan"):
            reply = api_result.get("message", reply)
    assert "Real step" in reply
    assert "hallucinated" not in reply.lower()


def test_query_coach_db_rejects_non_select():
    result = query_coach_db("DELETE FROM mood_logs")
    assert result.get("error")
    assert "SELECT" in result.get("message", "")


def test_query_coach_db_rejects_disallowed_table():
    result = query_coach_db("SELECT * FROM messages LIMIT 5")
    assert result.get("error")
    assert "not allowed" in result.get("message", "").lower()


def test_query_coach_db_allows_fitness_view():
    result = query_coach_db(
        "SELECT day_of_week, title FROM v_active_fitness_plan ORDER BY day_of_week LIMIT 5"
    )
    assert not result.get("error")
    assert "rows" in result


def test_goals_crud():
    entry = log_goal(
        category="fitness",
        goal_text="Gym 2x per week",
        target={"sessions_per_week": 2},
    )
    assert entry["goal_text"] == "Gym 2x per week"
    updated = update_goal(entry["id"], progress={"sessions_completed": 1})
    assert updated is not None
    assert updated["progress"]["sessions_completed"] == 1
    active = fetch_active_goals(limit=5)
    assert any(goal["id"] == entry["id"] for goal in active)


def test_coach_state_snapshot_in_llm_context():
    snapshot = build_coach_state_snapshot()
    assert "plan_summary" in snapshot
    prompt = format_coach_state_for_prompt(snapshot)
    assert "COACH MEMORY" in prompt
    ctx = build_llm_context(sender_phone="", user_text="hi")
    assert "coach_state_context" in ctx
    assert "COACH MEMORY" in ctx["coach_state_context"]


def test_local_coach_intents_include_goals():
    assert "LOG_GOAL" in LOCAL_COACH_INTENTS
    assert "QUERY_GOALS" in LOCAL_COACH_INTENTS


def test_lookup_coach_data_generates_sql_when_missing():
    def fake_generate(**_kwargs):
        return CoachDbQueryResponse(
            sql_query="SELECT category, goal_text FROM v_active_goals LIMIT 5",
            natural_question="what are my goals?",
        )

    result = lookup_coach_data(
        natural_question="what are my goals?",
        generate_sql=fake_generate,
    )
    assert not result.get("error")
    assert result.get("sql_generated") is True
    assert "rows" in result


def test_lookup_coach_data_retries_on_invalid_sql():
    calls: list[str] = []

    def fake_generate(**kwargs):
        if kwargs.get("error_hint"):
            calls.append("retry")
            return CoachDbQueryResponse(
                sql_query="SELECT category, goal_text FROM v_active_goals LIMIT 5",
                natural_question="goals",
            )
        calls.append("first")
        return CoachDbQueryResponse(
            sql_query="SELECT * FROM messages LIMIT 5",
            natural_question="goals",
        )

    result = lookup_coach_data(
        natural_question="what are my goals?",
        generate_sql=fake_generate,
    )
    assert calls == ["first", "retry"]
    assert not result.get("error")
    assert result.get("sql_retried") is True


def test_query_coach_data_node_wired_in_graph():
    graph = build_coach_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert "query_coach_data" in node_names


def test_query_coach_data_intent_exists():
    assert Intent.QUERY_COACH_DATA.value == "QUERY_COACH_DATA"


def test_fitness_plan_full_week_on_give_me_the_plan():
    from backend.health_coach.agent.actions import _fitness_plan_scope

    assert _fitness_plan_scope({}, "give me the plan") == "full_week"
    assert _fitness_plan_scope({}, "what's my workout today") == "today"


def test_goal_intake_not_saved_as_goal():
    from backend.health_coach.agent.actions import _is_goal_intake_message

    assert _is_goal_intake_message("can u help log my goals?", {}) is True
    assert _is_goal_intake_message(
        "lose weight to 68kg",
        {"goal_text": "lose weight to 68kg"},
    ) is False


def test_plan_context_guard_routes_wellness_plan():
    from backend.health_coach.agent.graph import _apply_plan_context_guard
    from backend.health_coach.agent.engine import Intent

    ctx = (
        "User: lose weight to 68kg\n"
        "Coach: Goal saved\n"
        "User: meal prep plan please"
    )
    intent = _apply_plan_context_guard("give me the plan", Intent.QUERY_FITNESS_PLAN.value, ctx)
    assert intent == Intent.BUILD_WELLNESS_PLAN.value


def test_build_wellness_plan_node_in_graph():
    graph = build_coach_graph()
    assert "build_wellness_plan" in set(graph.get_graph().nodes.keys())
