"""
Multi-agent LangGraph coach for the WhatsApp AI Health Coach.
"""

from __future__ import annotations

import logging
import operator
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send, interrupt

from ..core.health_normalizer import normalize_health_result
from ..core.payloads import expand_nutrition_items
from ..integrations.google_health import GoogleHealthClient
from ..integrations.nutrition import (
    compose_nutrition_reply,
    needs_nutrition_lookup,
    search_food_nutrition,
    should_skip_health_sync,
)
from ..integrations.research import search_health_topic
from ..services.coach_db_tool import lookup_coach_data
from ..services.coaching_preferences import detect_and_store_coaching_focus
from ..services.llm_context import build_llm_context
from ..services.memory import record_exchange
from ..services.wellness_plans import fetch_wellness_plan_context, save_wellness_plan_note
from .actions import QUERY_INTENTS, execute_health_action
from .engine import AIEngine, Intent
from .intent_registry import (
    LOCAL_COACH_INTENTS,
    get_capability,
    is_batch_nutrition,
    route_after_intent,
    route_after_nutrition_lookup,
)
from .vision import VisionAgent, _caption_requests_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPH_CHECKPOINT_PATH = os.getenv(
    "GRAPH_CHECKPOINT_PATH",
    str(PROJECT_ROOT / "data" / "coach_graph.sqlite3"),
)


class CoachState(TypedDict, total=False):
    user_text: str
    sender_phone: str
    message_type: str
    image_bytes: bytes | None
    image_mime_type: str | None
    image_caption: str
    document_bytes: bytes | None
    document_mime_type: str | None
    document_filename: str
    audio_bytes: bytes | None
    audio_mime_type: str | None
    intent: str
    payload: dict[str, Any]
    conversational_reply: str
    vision_result: dict[str, Any] | None
    nutrition_search_result: dict[str, Any] | None
    research_result: dict[str, Any] | None
    api_result: dict[str, Any] | None
    batch_results: Annotated[list[dict[str, Any]], operator.add]
    batch_item: dict[str, Any]
    final_reply: str
    pending_confirm: bool
    use_interactive_buttons: bool


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

_WELLNESS_TOPIC_SIGNALS = (
    "weight",
    "meal",
    "nutrition",
    "lose",
    "kilos",
    "kg",
    "meal prep",
    "calories",
    "alcohol",
    "diet",
    "food",
)
_FITNESS_TOPIC_SIGNALS = (
    "gym",
    "workout",
    "exercise",
    "fitness",
    "tuesday",
    "thursday",
    "tue",
    "thur",
    "sessions",
)


def _apply_no_log_guard(user_text: str, intent: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    lowered = user_text.lower()
    if not any(phrase in lowered for phrase in NO_LOG_PHRASES):
        return intent, payload
    if intent == Intent.LOG_NUTRITION.value:
        return Intent.QUERY_NUTRITION.value, payload
    if intent in {Intent.LOG_HYDRATION.value, Intent.LOG_WEIGHT.value, Intent.LOG_EXERCISE.value}:
        guarded = dict(payload)
        guarded.setdefault("needs_web_search", False)
        return Intent.COACHING_CHAT.value, guarded
    return intent, payload


def _apply_plan_context_guard(
    user_text: str,
    intent: str,
    conversation_context: str,
) -> str:
    if intent != Intent.QUERY_FITNESS_PLAN.value:
        return intent
    lowered = user_text.lower()
    if not any(phrase in lowered for phrase in ("the plan", "give me the plan", "my plan", "full plan")):
        return intent
    if any(signal in lowered for signal in _FITNESS_TOPIC_SIGNALS):
        return intent
    ctx = conversation_context.lower()
    wellness_score = sum(1 for signal in _WELLNESS_TOPIC_SIGNALS if signal in ctx)
    fitness_score = sum(1 for signal in _FITNESS_TOPIC_SIGNALS if signal in ctx)
    if wellness_score >= 2 and wellness_score >= fitness_score:
        return Intent.BUILD_WELLNESS_PLAN.value
    return intent


def _apply_wellness_plan_phrasing(user_text: str, intent: str) -> str:
    lowered = user_text.lower()
    if intent != Intent.COACHING_CHAT.value:
        return intent
    if any(
        phrase in lowered
        for phrase in (
            "build my meal",
            "meal and workout plan",
            "meal prep plan",
            "weight loss plan",
            "tailored plan",
            "nutrition plan",
        )
    ):
        return Intent.BUILD_WELLNESS_PLAN.value
    return intent


def _is_confirm_response(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"yes", "y", "confirm", "log it", "log", "ok", "okay", "confirm_log", "yes log it"}


def _is_skip_response(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"no", "n", "skip", "cancel", "don't log", "dont log", "skip_log"}


def _needs_nutrition_confirm(intent: str, payload: dict[str, Any]) -> bool:
    if intent != Intent.LOG_NUTRITION.value:
        return False
    if payload.get("from_image"):
        return True
    confidence = str(payload.get("nutrition_confidence", "")).lower()
    if confidence == "low":
        return True
    calories = payload.get("calories_kcal")
    try:
        if calories and float(calories) > 1200:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _compose_confirm_message(payload: dict[str, Any]) -> str:
    food = payload.get("food_display_name", "this item")
    kcal = payload.get("calories_kcal", "?")
    portion = payload.get("portion_description", "")
    portion_bit = f" ({portion})" if portion else ""
    return f"Log {food}{portion_bit} at ~{kcal} kcal to Google Health?"


_checkpointer_ctx = None
_checkpointer_instance = None


def _create_checkpointer():
    global _checkpointer_ctx, _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(GRAPH_CHECKPOINT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        _checkpointer_ctx = SqliteSaver.from_conn_string(str(path))
        _checkpointer_instance = _checkpointer_ctx.__enter__()
        return _checkpointer_instance
    except Exception as exc:
        logger.warning("SqliteSaver unavailable (%s); using in-memory checkpointer.", exc)
        from langgraph.checkpoint.memory import InMemorySaver

        _checkpointer_instance = InMemorySaver()
        return _checkpointer_instance


def build_coach_graph(
    *,
    ai_engine: AIEngine | None = None,
    health_client: GoogleHealthClient | None = None,
    checkpointer: Any | None = None,
):
    engine = ai_engine or AIEngine()
    client = health_client or GoogleHealthClient()
    vision = VisionAgent(client=engine._client)

    def _llm_context(state: CoachState) -> dict[str, str]:
        return build_llm_context(
            sender_phone=state.get("sender_phone", ""),
            user_text=state.get("user_text", "") or state.get("image_caption", ""),
            health_client=client,
        )

    def prepare_input(state: CoachState) -> CoachState:
        if state.get("message_type") == "audio" and state.get("audio_bytes"):
            transcript = engine.transcribe_audio(
                audio_bytes=state["audio_bytes"],
                mime_type=state.get("audio_mime_type") or "audio/ogg",
            )
            if not transcript:
                return {
                    "user_text": "",
                    "intent": Intent.COACHING_CHAT.value,
                    "payload": {},
                    "conversational_reply": (
                        "I couldn't make out that voice note. Try again or type your message."
                    ),
                }
            return {"user_text": transcript}
        if state.get("message_type") == "document" and state.get("document_bytes"):
            ctx = _llm_context(state)
            summary = engine.summarize_document(
                document_bytes=state["document_bytes"],
                mime_type=state.get("document_mime_type") or "application/pdf",
                filename=state.get("document_filename") or "document",
                user_question=state.get("user_text") or state.get("image_caption", ""),
                conversation_context=ctx["conversation_context"],
                user_profile_context=ctx["user_profile_context"],
            )
            return {
                "intent": Intent.SUMMARIZE_DOCUMENT.value,
                "payload": {"filename": state.get("document_filename")},
                "conversational_reply": summary,
                "final_reply": summary,
            }
        return {}

    def analyze_food_image(state: CoachState) -> CoachState:
        ctx = _llm_context(state)
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
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
        )
        intent = Intent.QUERY_NUTRITION.value
        if _caption_requests_logging(caption) or (
            vision_result.get("wants_to_log") and not vision_result.get("lookup_only")
        ):
            intent = Intent.LOG_NUTRITION.value

        payload = {
            "food_display_name": vision_result.get("food_display_name", "Meal from photo"),
            "portion_description": vision_result.get("portion_description", "1 serving"),
            "meal_type": vision_result.get("meal_type", "MEAL_TYPE_UNSPECIFIED"),
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
            "use_interactive_buttons": intent == Intent.LOG_NUTRITION.value,
            "api_result": None,
        }

    def route_intent(state: CoachState) -> CoachState:
        if state.get("final_reply"):
            return {}
        user_text = state.get("user_text", "")
        detect_and_store_coaching_focus(user_text)
        ctx = _llm_context(state)
        routed = engine.route_message(
            user_text,
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
            coach_state_context=ctx.get("coach_state_context", ""),
        )
        intent, payload = _apply_no_log_guard(user_text, routed.intent.value, routed.payload)
        intent = _apply_plan_context_guard(user_text, intent, ctx["conversation_context"])
        intent = _apply_wellness_plan_phrasing(user_text, intent)
        return {
            "intent": intent,
            "payload": payload,
            "conversational_reply": routed.conversational_reply,
            "api_result": None,
        }

    def lookup_nutrition(state: CoachState) -> CoachState | Command:
        ctx = _llm_context(state)
        payload = dict(state.get("payload", {}))
        item = state.get("batch_item") or payload
        search_result = search_food_nutrition(
            food_display_name=item.get("food_display_name", "meal"),
            portion_description=item.get("portion_description", ""),
            user_message=state["user_text"],
        )
        enriched_payload = engine.resolve_nutrition_macros(
            user_text=state["user_text"],
            payload=item,
            search_result=search_result,
            intent=state.get("intent", "LOG_NUTRITION"),
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
        )
        reply = compose_nutrition_reply(
            base_reply=state.get("conversational_reply", "") if not state.get("batch_item") else "",
            resolved=enriched_payload,
        )
        intent = state.get("intent", Intent.LOG_NUTRITION.value)
        if state.get("batch_item"):
            item_reply = compose_nutrition_reply(base_reply="", resolved=enriched_payload)
            if intent == Intent.QUERY_NUTRITION.value or should_skip_health_sync(intent, enriched_payload):
                return {"batch_results": [{"item": enriched_payload, "synced": False, "reply": item_reply}]}
            api_result = execute_health_action(
                intent,
                enriched_payload,
                client=client,
                user_text=state.get("user_text", ""),
            )
            return {"batch_results": [{"item": enriched_payload, "api_result": api_result, "reply": item_reply}]}

        if _needs_nutrition_confirm(intent, enriched_payload):
            choice = interrupt(
                {
                    "reply": _compose_confirm_message(enriched_payload),
                    "payload": enriched_payload,
                    "use_buttons": state.get("use_interactive_buttons", False),
                }
            )
            if isinstance(choice, dict):
                choice_text = str(choice.get("choice", choice.get("text", "")))
            else:
                choice_text = str(choice)
            if _is_skip_response(choice_text):
                return Command(
                    update={
                        "payload": enriched_payload,
                        "nutrition_search_result": search_result,
                        "conversational_reply": reply,
                        "final_reply": f"Skipped logging. {reply}".strip(),
                    },
                    goto=END,
                )
            if not _is_confirm_response(choice_text):
                return Command(
                    update={
                        "payload": enriched_payload,
                        "nutrition_search_result": search_result,
                        "conversational_reply": reply,
                        "final_reply": (
                            f"{reply}\n\nReply 'yes' to log or 'skip' to cancel."
                        ).strip(),
                        "pending_confirm": True,
                    },
                    goto=END,
                )

        return {
            "payload": enriched_payload,
            "nutrition_search_result": search_result,
            "conversational_reply": reply,
        }

    def process_batch_item(state: CoachState) -> CoachState:
        ctx = _llm_context(state)
        item = state.get("batch_item", {})
        intent = state.get("intent", Intent.LOG_NUTRITION.value)
        search_result = search_food_nutrition(
            food_display_name=item.get("food_display_name", "meal"),
            portion_description=item.get("portion_description", ""),
            user_message=state["user_text"],
        )
        enriched = engine.resolve_nutrition_macros(
            user_text=state["user_text"],
            payload=item,
            search_result=search_result,
            intent=intent,
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
        )
        item_reply = compose_nutrition_reply(base_reply="", resolved=enriched)
        if intent == Intent.QUERY_NUTRITION.value or should_skip_health_sync(intent, enriched):
            return {"batch_results": [{"item": enriched, "synced": False, "reply": item_reply}]}
        api_result = execute_health_action(
            intent,
            enriched,
            client=client,
            user_text=state.get("user_text", ""),
        )
        return {"batch_results": [{"item": enriched, "api_result": api_result, "reply": item_reply}]}

    def batch_log_nutrition(state: CoachState) -> CoachState:
        ctx = _llm_context(state)
        items = expand_nutrition_items(state.get("payload", {}))
        intent = state.get("intent", Intent.LOG_NUTRITION.value)
        lines: list[str] = []
        if state.get("conversational_reply"):
            lines.append(state["conversational_reply"].strip())
        batch_results: list[dict[str, Any]] = list(state.get("batch_results") or [])
        api_errors: list[str] = []

        if not batch_results:
            for item in items:
                search_result = search_food_nutrition(
                    food_display_name=item.get("food_display_name", "meal"),
                    portion_description=item.get("portion_description", ""),
                    user_message=state["user_text"],
                )
                enriched = engine.resolve_nutrition_macros(
                    user_text=state["user_text"],
                    payload=item,
                    search_result=search_result,
                    intent=intent,
                    conversation_context=ctx["conversation_context"],
                    user_profile_context=ctx["user_profile_context"],
                )
                item_reply = compose_nutrition_reply(base_reply="", resolved=enriched)
                if intent == Intent.QUERY_NUTRITION.value or should_skip_health_sync(intent, enriched):
                    lines.append(item_reply)
                    batch_results.append({"item": enriched, "synced": False, "reply": item_reply})
                    continue
                api_result = execute_health_action(
                    intent,
                    enriched,
                    client=client,
                    user_text=state.get("user_text", ""),
                )
                batch_results.append({"item": enriched, "api_result": api_result, "reply": item_reply})
                if api_result and api_result.get("error"):
                    api_errors.append(
                        f"{enriched.get('food_display_name')}: {api_result.get('message', 'sync failed')}"
                    )
                else:
                    lines.append(item_reply)
        else:
            for entry in batch_results:
                item_reply = entry.get("reply", "")
                api_result = entry.get("api_result")
                if api_result and api_result.get("error"):
                    api_errors.append(
                        f"{entry.get('item', {}).get('food_display_name')}: "
                        f"{api_result.get('message', 'sync failed')}"
                    )
                elif item_reply:
                    lines.append(item_reply)

        reply = "\n\n".join(line for line in lines if line.strip())
        if api_errors:
            reply += "\n\n(Some items could not sync: " + "; ".join(api_errors) + ")"
        return {
            "batch_results": batch_results,
            "conversational_reply": reply,
            "final_reply": reply,
        }

    def research_answer(state: CoachState) -> CoachState:
        ctx = _llm_context(state)
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
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
        )
        return {"research_result": search_result, "final_reply": reply}

    def execute_health(state: CoachState) -> CoachState:
        result = execute_health_action(
            state["intent"],
            state.get("payload", {}),
            client=client,
            sender_phone=state.get("sender_phone"),
            user_text=state.get("user_text", ""),
        )
        return {"api_result": result}

    def query_coach_data(state: CoachState) -> CoachState:
        ctx = _llm_context(state)
        payload = state.get("payload", {})
        natural_question = payload.get("natural_question") or state.get("user_text", "")
        sql_query = (payload.get("sql_query") or "").strip()

        def _generate_sql(**kwargs: Any) -> Any:
            return engine.generate_coach_db_query(
                user_text=state["user_text"],
                natural_question=kwargs.get("natural_question") or natural_question,
                conversation_context=ctx["conversation_context"],
                coach_state_context=ctx.get("coach_state_context", ""),
                error_hint=kwargs.get("error_hint", ""),
                previous_sql=kwargs.get("previous_sql", ""),
            )

        query_result = lookup_coach_data(
            natural_question=natural_question,
            sql_query=sql_query,
            generate_sql=_generate_sql,
        )
        if query_result.get("error"):
            return {
                "api_result": query_result,
                "final_reply": query_result.get(
                    "message",
                    "I couldn't look that up in your coach history right now.",
                ),
            }
        reply = engine.summarize_coach_data(
            user_text=state["user_text"],
            draft_reply=state.get("conversational_reply", ""),
            query_result=query_result,
            natural_question=natural_question,
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
            coach_state_context=ctx.get("coach_state_context", ""),
        )
        return {"api_result": query_result, "final_reply": reply}

    def build_wellness_plan(state: CoachState) -> CoachState:
        ctx = _llm_context(state)
        payload = state.get("payload", {})
        lookback = int(payload.get("lookback_days") or 21)
        wellness_context = fetch_wellness_plan_context(
            client=client,
            lookback_days=lookback,
        )
        generated = engine.generate_wellness_plan(
            user_text=state["user_text"],
            payload=payload,
            wellness_context=wellness_context,
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
            coach_state_context=ctx.get("coach_state_context", ""),
        )
        if generated is None:
            return {
                "final_reply": (
                    "I couldn't build your wellness plan right now. "
                    "Try again in a moment, or ask for your fitness plan separately."
                ),
            }
        save_wellness_plan_note(message=generated.final_reply, context=wellness_context)
        return {
            "api_result": {"wellness_plan": generated.model_dump(), "context": wellness_context},
            "final_reply": generated.final_reply,
        }

    def finalize_reply(state: CoachState) -> CoachState:
        if state.get("final_reply"):
            return {}
        ctx = _llm_context(state)
        reply = state.get("conversational_reply", "")
        api_result = state.get("api_result")
        intent_name = state.get("intent", "")

        if api_result and api_result.get("message") and not api_result.get("error"):
            if intent_name in LOCAL_COACH_INTENTS or api_result.get("plan"):
                reply = api_result.get("message", reply)
            elif api_result.get("today_workout") or api_result.get("entries") is not None:
                reply = api_result.get("message", reply)

        if api_result and api_result.get("error"):
            reply = (
                f"{reply}\n\n"
                f"(Heads up: I couldn't sync with Google Health just now — "
                f"{api_result.get('message', 'please try again shortly')})"
            )
            return {"final_reply": reply.strip()}

        if api_result and api_result.get("replacement_strategy"):
            note = api_result.get("message", "")
            if note:
                reply = f"{reply}\n\n{note}"
            return {"final_reply": reply.strip()}

        retry_meta = (api_result or {}).get("_health_sync_retry")
        if retry_meta and retry_meta.get("applied"):
            summary = retry_meta.get("fix_summary", "I corrected the log format and synced successfully.")
            if summary and summary not in reply:
                reply = f"{reply}\n\n({summary})".strip()

        try:
            intent = Intent(intent_name)
        except ValueError:
            return {"final_reply": reply.strip()}

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
                conversation_context=ctx["conversation_context"],
                user_profile_context=ctx["user_profile_context"],
            )

        return {"final_reply": reply.strip()}

    def entry_route(
        state: CoachState,
    ) -> Literal["prepare_input", "analyze_food_image", "route_intent"]:
        if state.get("message_type") in {"audio", "document"}:
            return "prepare_input"
        if state.get("message_type") == "image" and state.get("image_bytes"):
            return "analyze_food_image"
        return "route_intent"

    def after_prepare(
        state: CoachState,
    ) -> Literal["route_intent", "finalize_reply"]:
        if state.get("final_reply"):
            return "finalize_reply"
        return "route_intent"

    def after_intent_route(state: CoachState) -> str | list[Send]:
        intent = state.get("intent", "")
        payload = state.get("payload", {})
        if is_batch_nutrition(payload) and get_capability(intent).supports_batch:
            items = expand_nutrition_items(payload)
            return [
                Send(
                    "process_batch_item",
                    {
                        **state,
                        "batch_item": item,
                        "batch_results": [],
                    },
                )
                for item in items
            ]
        return route_after_intent(intent, payload)

    def after_vision(state: CoachState) -> str:
        return route_after_intent(state.get("intent", ""), state.get("payload", {}))

    def after_nutrition_lookup(state: CoachState) -> str:
        return route_after_nutrition_lookup(
            state.get("intent", ""),
            state.get("payload", {}),
        )

    workflow = StateGraph(CoachState)
    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("analyze_food_image", analyze_food_image)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("lookup_nutrition", lookup_nutrition)
    workflow.add_node("process_batch_item", process_batch_item)
    workflow.add_node("batch_log_nutrition", batch_log_nutrition)
    workflow.add_node("research_answer", research_answer)
    workflow.add_node("execute_health", execute_health)
    workflow.add_node("query_coach_data", query_coach_data)
    workflow.add_node("build_wellness_plan", build_wellness_plan)
    workflow.add_node("finalize_reply", finalize_reply)

    workflow.set_conditional_entry_point(
        entry_route,
        {
            "prepare_input": "prepare_input",
            "analyze_food_image": "analyze_food_image",
            "route_intent": "route_intent",
        },
    )
    workflow.add_conditional_edges(
        "prepare_input",
        after_prepare,
        {"route_intent": "route_intent", "finalize_reply": "finalize_reply"},
    )
    workflow.add_conditional_edges(
        "analyze_food_image",
        after_vision,
        {
            "batch_log_nutrition": "batch_log_nutrition",
            "lookup_nutrition": "lookup_nutrition",
            "research_answer": "research_answer",
            "execute_health": "execute_health",
            "query_coach_data": "query_coach_data",
            "build_wellness_plan": "build_wellness_plan",
            "finalize_reply": "finalize_reply",
        },
    )
    workflow.add_conditional_edges(
        "route_intent",
        after_intent_route,
        {
            "batch_log_nutrition": "batch_log_nutrition",
            "lookup_nutrition": "lookup_nutrition",
            "research_answer": "research_answer",
            "execute_health": "execute_health",
            "query_coach_data": "query_coach_data",
            "build_wellness_plan": "build_wellness_plan",
            "finalize_reply": "finalize_reply",
            "process_batch_item": "process_batch_item",
        },
    )
    workflow.add_edge("process_batch_item", "batch_log_nutrition")
    workflow.add_conditional_edges(
        "lookup_nutrition",
        after_nutrition_lookup,
        {
            "execute_health": "execute_health",
            "finalize_reply": "finalize_reply",
        },
    )
    workflow.add_edge("batch_log_nutrition", END)
    workflow.add_edge("research_answer", END)
    workflow.add_edge("query_coach_data", END)
    workflow.add_edge("build_wellness_plan", END)
    workflow.add_edge("execute_health", "finalize_reply")
    workflow.add_edge("finalize_reply", END)

    saver = checkpointer if checkpointer is not None else _create_checkpointer()
    return workflow.compile(checkpointer=saver)


_coach_graph = None


def get_coach_graph():
    global _coach_graph
    if _coach_graph is None:
        _coach_graph = build_coach_graph()
    return _coach_graph


def _graph_config(sender_phone: str) -> dict[str, Any]:
    thread_id = sender_phone or "default"
    return {"configurable": {"thread_id": thread_id}}


def _has_pending_interrupt(graph, config: dict[str, Any]) -> bool:
    try:
        state = graph.get_state(config)
        return bool(getattr(state, "tasks", None) or getattr(state, "next", None))
    except Exception:
        return False


def run_coach(
    user_text: str = "",
    sender_phone: str = "",
    *,
    message_type: str = "text",
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    image_caption: str = "",
    document_bytes: bytes | None = None,
    document_mime_type: str | None = None,
    document_filename: str = "",
    audio_bytes: bytes | None = None,
    audio_mime_type: str | None = None,
) -> CoachState:
    graph = get_coach_graph()
    config = _graph_config(sender_phone)

    if sender_phone and _has_pending_interrupt(graph, config):
        if _is_confirm_response(user_text) or _is_skip_response(user_text):
            result = graph.invoke(Command(resume=user_text), config=config)
        else:
            result = graph.invoke(Command(resume=user_text), config=config)
    else:
        result = graph.invoke(
            {
                "user_text": user_text,
                "sender_phone": sender_phone,
                "message_type": message_type,
                "image_bytes": image_bytes,
                "image_mime_type": image_mime_type,
                "image_caption": image_caption,
                "document_bytes": document_bytes,
                "document_mime_type": document_mime_type,
                "document_filename": document_filename,
                "audio_bytes": audio_bytes,
                "audio_mime_type": audio_mime_type,
                "batch_results": [],
            },
            config=config,
        )

    if isinstance(result, dict):
        coach_state = result
    else:
        coach_state = dict(result)

    reply = coach_state.get("final_reply") or coach_state.get("conversational_reply", "")
    memory_text = user_text or image_caption or document_filename or "Sent media"
    if sender_phone and reply:
        record_exchange(sender_phone, user_text=memory_text, coach_reply=reply)
    return coach_state
