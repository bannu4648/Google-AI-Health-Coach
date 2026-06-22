from backend.health_coach.agent.graph import _apply_workout_followup_guard
from backend.health_coach.agent.engine import Intent
from backend.health_coach.integrations.nutrition import needs_nutrition_lookup
from backend.health_coach.services.memory import format_history_for_prompt
from backend.health_coach.services.pending_actions import (
    clear_pending_nutrition,
    is_log_followup_text,
    load_pending_nutrition,
    save_pending_nutrition,
)


def test_is_log_followup_text():
    assert is_log_followup_text("log it")
    assert is_log_followup_text("can u pls log that pasta")
    assert not is_log_followup_text("the gym is closed")


def test_pending_nutrition_roundtrip(tmp_path, monkeypatch):
    db_path = tmp_path / "coach.sqlite3"
    monkeypatch.setenv("HEALTH_COACH_DB_PATH", str(db_path))
    payload = {"food_display_name": "linguine", "calories_kcal": 467}
    save_pending_nutrition("85253016865", payload=payload, user_text="photo")
    loaded = load_pending_nutrition("85253016865")
    assert loaded["payload"]["food_display_name"] == "linguine"
    clear_pending_nutrition("85253016865")
    assert load_pending_nutrition("85253016865") is None


def test_needs_nutrition_lookup_skips_resolved_macros():
    resolved = {
        "food_display_name": "linguine",
        "calories_kcal": 467,
        "nutrition_resolution": "use_search",
    }
    assert not needs_nutrition_lookup("LOG_NUTRITION", resolved)
    # Router shorthand calories without a completed resolution still run Tavily.
    assert needs_nutrition_lookup(
        "LOG_NUTRITION",
        {"food_display_name": "linguine", "calories_kcal": 467},
    )


def test_workout_followup_guard_after_scheduled_nudge():
    ctx = "Coach (scheduled): Workout reminder: Full body gym is on your plan today."
    intent = _apply_workout_followup_guard(
        "the gym is closed! maybe give me suggestions that i can do indoors after dinner",
        Intent.QUERY_NUTRITION.value,
        ctx,
    )
    assert intent == Intent.COACHING_CHAT.value


def test_scheduled_message_label_in_history(monkeypatch):
    monkeypatch.setattr(
        "backend.health_coach.services.memory.fetch_recent_messages_for_phone",
        lambda phone, limit: [
            {
                "direction": "outbound",
                "text": "Workout reminder: Full body gym is on your plan today.",
            },
            {"direction": "inbound", "text": "gym is closed"},
        ],
    )
    history = format_history_for_prompt("85253016865", intent="COACHING_CHAT")
    assert "Coach (scheduled)" in history
    assert "Workout reminder" in history


def test_log_followup_clears_stale_final_reply():
    from backend.health_coach.agent.graph import build_coach_graph, _graph_config, _build_invoke_input
    from backend.health_coach.services.pending_actions import save_pending_nutrition, clear_pending_nutrition
    from langgraph.checkpoint.memory import InMemorySaver

    phone = "test_stale_final"
    payload = {"food_display_name": "pasta", "calories_kcal": 400, "meal_type": "LUNCH"}
    save_pending_nutrition(phone, payload=payload)
    graph = build_coach_graph(checkpointer=InMemorySaver())
    config = _graph_config(phone)

    graph.invoke(
        {
            **_build_invoke_input(user_text="photo pasta", sender_phone=phone, message_type="text"),
            "final_reply": "Old nutrition lookup reply that should not block logging.",
            "intent": "QUERY_NUTRITION",
            "conversational_reply": "Old nutrition lookup reply that should not block logging.",
        },
        config=config,
    )
    result = graph.invoke(
        _build_invoke_input(user_text="log it", sender_phone=phone, message_type="text"),
        config=config,
    )
    clear_pending_nutrition(phone)
    assert result.get("intent") == "LOG_NUTRITION"
    assert "Old nutrition lookup reply" not in (result.get("final_reply") or "")
