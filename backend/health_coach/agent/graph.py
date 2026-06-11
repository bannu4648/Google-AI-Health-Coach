"""
Multi-agent LangGraph coach for the WhatsApp AI Health Coach.

Agents:
  - Vision agent (food photos)
  - Router agent (text intent)
  - Nutrition lookup agent (Tavily + macro resolution)
  - Research agent (general sourced questions)
  - Health sync agent (Google Health API)
  - Summarizer agent (query replies)

Flow:
  [image] analyze_food_image -> lookup_nutrition -> execute_health | finalize_reply
  [text]  route_intent -> lookup_nutrition | research_answer | execute_health | finalize_reply
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
from .vision import VisionAgent, _caption_requests_logging

logger = logging.getLogger(__name__)


class CoachState(TypedDict, total=False):
    user_text: str
    sender_phone: str
    message_type: str
    image_bytes: bytes | None
    image_mime_type: str | None
    image_caption: str
    intent: str
    payload: dict[str, Any]
    conversational_reply: str
    vision_result: dict[str, Any] | None
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
    vision = VisionAgent(client=engine._client)

    def analyze_food_image(state: CoachState) -> CoachState:
        history = format_history_for_prompt(state.get("sender_phone", ""))
        image_bytes = state.get("image_bytes")
        mime_type = state.get("image_mime_type") or "image/jpeg"
        caption = state.get("image_caption", "")
        if not image_bytes:
            return {
                "intent": Intent.COACHING_CHAT.value,
                "payload": {},
                "conversational_reply": (
                    "I couldn't read that photo. Try sending the image again, "
                    "or describe the meal in text."
                ),
            }

        vision_result = vision.analyze_food_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            caption=caption,
            conversation_context=history,
        )
        # Photos default to nutrition lookup only; log only on explicit caption or vision flag.
        intent = Intent.QUERY_NUTRITION.value
        if _caption_requests_logging(caption) or (
            vision_result.get("wants_to_log") and not vision_result.get("lookup_only")
        ):
            intent = Intent.LOG_NUTRITION.value

        payload = {
            "food_display_name": vision_result.get("food_display_name", "Meal from photo"),
            "portion_description": vision_result.get("portion_description", "1 serving"),
            "meal_type": vision_result.get("meal_type", "UNKNOWN"),
            "vision_notes": vision_result.get("vision_notes", ""),
            "from_image": True,
        }
        user_text = caption.strip() or f"Photo: {payload['food_display_name']}"
        return {
            "user_text": user_text,
            "intent": intent,
            "payload": payload,
            "vision_result": vision_result,
            "conversational_reply": vision_result.get("conversational_reply", ""),
            "api_result": None,
        }

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

    def entry_route(
        state: CoachState,
    ) -> Literal["analyze_food_image", "route_intent"]:
        if state.get("message_type") == "image" and state.get("image_bytes"):
            return "analyze_food_image"
        return "route_intent"

    def after_vision(
        state: CoachState,
    ) -> Literal["lookup_nutrition", "finalize_reply"]:
        if needs_nutrition_lookup(state.get("intent", ""), state.get("payload", {})):
            return "lookup_nutrition"
        return "finalize_reply"

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
    workflow.add_node("analyze_food_image", analyze_food_image)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("lookup_nutrition", lookup_nutrition)
    workflow.add_node("research_answer", research_answer)
    workflow.add_node("execute_health", execute_health)
    workflow.add_node("finalize_reply", finalize_reply)

    workflow.set_conditional_entry_point(
        entry_route,
        {
            "analyze_food_image": "analyze_food_image",
            "route_intent": "route_intent",
        },
    )
    workflow.add_conditional_edges(
        "analyze_food_image",
        after_vision,
        {
            "lookup_nutrition": "lookup_nutrition",
            "finalize_reply": "finalize_reply",
        },
    )
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


_coach_graph = None


def get_coach_graph():
    """Lazy singleton so imports work before GEMINI_API_KEY is loaded in tests."""
    global _coach_graph
    if _coach_graph is None:
        _coach_graph = build_coach_graph()
    return _coach_graph


def run_coach(
    user_text: str = "",
    sender_phone: str = "",
    *,
    message_type: str = "text",
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    image_caption: str = "",
) -> CoachState:
    """Invoke the compiled multi-agent graph for a single inbound message."""
    result = get_coach_graph().invoke(
        {
            "user_text": user_text,
            "sender_phone": sender_phone,
            "message_type": message_type,
            "image_bytes": image_bytes,
            "image_mime_type": image_mime_type,
            "image_caption": image_caption,
        }
    )
    reply = result.get("final_reply") or result.get("conversational_reply", "")
    memory_text = user_text or image_caption or "Sent a meal photo"
    if sender_phone and reply:
        record_exchange(sender_phone, user_text=memory_text, coach_reply=reply)
    return result
