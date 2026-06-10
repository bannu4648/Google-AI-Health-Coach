"""
LangGraph agent for the WhatsApp AI Health Coach.

Flow:
  route_intent -> [lookup_nutrition | execute_health | finalize_reply]
  lookup_nutrition -> [execute_health | finalize_reply] (skip sync on ask_followup)
  execute_health -> finalize_reply (queries summarized) -> END
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from ..core.health_normalizer import normalize_health_result
from ..integrations.google_health import GoogleHealthClient
from ..integrations.nutrition import (
    compose_nutrition_reply,
    needs_nutrition_lookup,
    search_food_nutrition,
    should_skip_health_sync,
)
from ..integrations.research import search_health_topic
from ..services.memory import format_history_for_prompt, record_exchange
from .actions import QUERY_INTENTS, execute_health_action
from .engine import AIEngine, Intent

logger = logging.getLogger(__name__)


class CoachState(TypedDict, total=False):
    user_text: str
    sender_phone: str
    intent: str
    payload: dict[str, Any]
    conversational_reply: str
    nutrition_search_result: dict[str, Any] | None
    research_result: dict[str, Any] | None
    api_result: dict[str, Any] | None
    final_reply: str


NO_LOG_PHRASES = (
    "don't log",
    "do not log",
    "dont log",
    "don't save",
    "do not save",
    "just curious",
    "lookup only",
    "don't add",
    "do not add",
)


def _apply_no_log_guard(user_text: str, intent: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Prevent accidental writes when the user explicitly asks not to log."""
    lowered = user_text.lower()
    if not any(phrase in lowered for phrase in NO_LOG_PHRASES):
        return intent, payload
    if intent == Intent.LOG_NUTRITION.value:
        return Intent.QUERY_NUTRITION.value, payload
    if intent in {Intent.LOG_HYDRATION.value, Intent.LOG_WEIGHT.value}:
        guarded = dict(payload)
        guarded.setdefault("needs_web_search", False)
        return Intent.COACHING_CHAT.value, guarded
    return intent, payload


def build_coach_graph(
    *,
    ai_engine: AIEngine | None = None,
    health_client: GoogleHealthClient | None = None,
):
    """Compile the LangGraph coach agent."""
    engine = ai_engine or AIEngine()
    client = health_client or GoogleHealthClient()

    def route_intent(state: CoachState) -> CoachState:
        history = format_history_for_prompt(state.get("sender_phone", ""))
        routed = engine.route_message(
            state["user_text"],
            conversation_context=history,
        )
        intent, payload = _apply_no_log_guard(
            state["user_text"],
            routed.intent.value,
            routed.payload,
        )
        return {
            "intent": intent,
            "payload": payload,
            "conversational_reply": routed.conversational_reply,
            "api_result": None,
        }

    def lookup_nutrition(state: CoachState) -> CoachState:
        payload = dict(state.get("payload", {}))
        search_result = search_food_nutrition(
            food_display_name=payload.get("food_display_name", "meal"),
            portion_description=payload.get("portion_description", ""),
            user_message=state["user_text"],
        )
        enriched_payload = engine.resolve_nutrition_macros(
            user_text=state["user_text"],
            payload=payload,
            search_result=search_result,
            intent=state.get("intent", "LOG_NUTRITION"),
        )
        reply = compose_nutrition_reply(
            base_reply=state.get("conversational_reply", ""),
            resolved=enriched_payload,
        )
        return {
            "payload": enriched_payload,
            "nutrition_search_result": search_result,
            "conversational_reply": reply,
        }

    def research_answer(state: CoachState) -> CoachState:
        payload = state.get("payload", {})
        query = (
            payload.get("search_query")
            or " ".join(payload.get("topics", []))
            or state["user_text"]
        )
        search_result = search_health_topic(query, user_message=state["user_text"])
        reply = engine.answer_research_question(
            user_text=state["user_text"],
            draft_reply=state.get("conversational_reply", ""),
            search_result=search_result,
        )
        return {"research_result": search_result, "final_reply": reply}

    def execute_health(state: CoachState) -> CoachState:
        result = execute_health_action(
            state["intent"],
            state.get("payload", {}),
            client=client,
        )
        return {"api_result": result}

    def finalize_reply(state: CoachState) -> CoachState:
        reply = state.get("conversational_reply", "")
        api_result = state.get("api_result")

        if api_result and api_result.get("error"):
            reply = (
                f"{reply}\n\n"
                f"(Heads up: I couldn't sync with Google Health just now — "
                f"{api_result.get('message', 'please try again shortly')})"
            )
            return {"final_reply": reply}

        if api_result and api_result.get("replacement_strategy"):
            note = api_result.get("message", "")
            if note:
                reply = f"{reply}\n\n{note}"
            return {"final_reply": reply}

        intent = Intent(state["intent"])
        if intent in QUERY_INTENTS and api_result is not None:
            data_type = state.get("payload", {}).get("data_type")
            normalized_result = (
                normalize_health_result(data_type, api_result)
                if data_type
                else api_result
            )
            reply = engine.summarize_health_data(
                user_text=state["user_text"],
                draft_reply=reply,
                api_result=normalized_result,
            )

        return {"final_reply": reply}

    def after_route(
        state: CoachState,
    ) -> Literal["lookup_nutrition", "research_answer", "execute_health", "finalize_reply"]:
        if state.get("intent") == Intent.GENERAL_RESEARCH.value:
            return "research_answer"
        if state.get("intent") == Intent.COACHING_CHAT.value:
            return "finalize_reply"
        if needs_nutrition_lookup(state.get("intent", ""), state.get("payload", {})):
            return "lookup_nutrition"
        return "execute_health"

    def after_nutrition_lookup(
        state: CoachState,
    ) -> Literal["execute_health", "finalize_reply"]:
        if should_skip_health_sync(
            state.get("intent", ""),
            state.get("payload", {}),
        ):
            return "finalize_reply"
        return "execute_health"

    workflow = StateGraph(CoachState)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("lookup_nutrition", lookup_nutrition)
    workflow.add_node("research_answer", research_answer)
    workflow.add_node("execute_health", execute_health)
    workflow.add_node("finalize_reply", finalize_reply)

    workflow.set_entry_point("route_intent")
    workflow.add_conditional_edges(
        "route_intent",
        after_route,
        {
            "lookup_nutrition": "lookup_nutrition",
            "research_answer": "research_answer",
            "execute_health": "execute_health",
            "finalize_reply": "finalize_reply",
        },
    )
    workflow.add_conditional_edges(
        "lookup_nutrition",
        after_nutrition_lookup,
        {
            "execute_health": "execute_health",
            "finalize_reply": "finalize_reply",
        },
    )
    workflow.add_edge("research_answer", END)
    workflow.add_edge("execute_health", "finalize_reply")
    workflow.add_edge("finalize_reply", END)

    return workflow.compile()


# Singleton graph used by the FastAPI webhook.
coach_graph = build_coach_graph()


def run_coach(user_text: str, sender_phone: str = "") -> CoachState:
    """Invoke the compiled graph for a single inbound message."""
    result = coach_graph.invoke(
        {
            "user_text": user_text,
            "sender_phone": sender_phone,
        }
    )
    reply = result.get("final_reply") or result.get("conversational_reply", "")
    if sender_phone and reply:
        record_exchange(sender_phone, user_text=user_text, coach_reply=reply)
    return result
