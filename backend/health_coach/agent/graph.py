"""
Multi-agent LangGraph coach for the WhatsApp AI Health Coach.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send, interrupt

from ..core.health_normalizer import normalize_health_result
from ..core.payloads import expand_nutrition_items
from ..integrations.google_auth import GoogleAuthRequiredError
from ..integrations.google_health import GoogleHealthClient
from ..integrations.exercise import compose_exercise_reply
from ..integrations.nutrition import (
    compose_nutrition_reply,
    needs_nutrition_lookup,
    search_food_nutrition,
    should_skip_health_sync,
)
from ..integrations.research import search_health_topic
from ..services.coach_db_tool import lookup_coach_data
from ..services.coaching_preferences import detect_and_store_coaching_focus
from ..services.pending_actions import (
    clear_pending_nutrition,
    is_log_followup_text,
    load_pending_nutrition,
    save_pending_nutrition,
)
from ..services.portion_hints import apply_caption_portion_hints
from ..services.exercise_resolver import resolve_exercise_payload_for_log
from ..services.llm_context import build_llm_context
from ..services.memory import record_exchange
from ..services.wellness_plans import fetch_wellness_plan_context, save_wellness_plan_note
from .actions import QUERY_INTENTS, execute_health_action
from .engine import AIEngine, Intent
from .intent_registry import (
    LOCAL_COACH_INTENTS,
    get_capability,
    is_batch_nutrition,
    route_after_exercise_lookup,
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


def _merge_batch_results(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Accumulate parallel batch items; an empty list clears stale checkpoint data."""
    if new is None:
        return list(existing or [])
    if len(new) == 0:
        return []
    return list(existing or []) + list(new)


def _batch_result_entry(
    *,
    item: dict[str, Any],
    reply: str,
    api_result: dict[str, Any] | None = None,
    synced: bool = True,
) -> dict[str, Any]:
    """Checkpoint-safe batch row (no non-serializable API client objects)."""
    entry: dict[str, Any] = {
        "item": item,
        "reply": reply,
        "synced": synced,
    }
    if api_result and api_result.get("error"):
        entry["sync_error"] = str(api_result.get("message") or "sync failed")
    return entry


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
    batch_results: Annotated[list[dict[str, Any]], _merge_batch_results]
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


def _is_day_review_request(user_text: str) -> bool:
    lowered = user_text.lower()
    time_refs = ("yesterday", "yday", "last night", "previous day", "today", "this morning")
    data_refs = ("food", "meal", "meals", "ate", "exercise", "exercises", "workout", "workouts", "logged")
    review_refs = (
        "healthy",
        "goal",
        "goals",
        "towards",
        "on track",
        "how did",
        "assessment",
        "good day",
        "tell me if",
        "was it",
    )
    has_time = any(token in lowered for token in time_refs)
    has_food = any(token in lowered for token in ("food", "meal", "meals", "ate", "nutrition"))
    has_exercise = any(token in lowered for token in ("exercise", "exercises", "workout", "workouts", "gym"))
    has_review = any(token in lowered for token in review_refs)
    return has_time and has_food and has_exercise and has_review


def _apply_day_review_guard(user_text: str, intent: str) -> tuple[str, dict]:
    if not _is_day_review_request(user_text):
        return intent, {}
    lowered = user_text.lower()
    day_offset = 0 if any(token in lowered for token in ("today", "this morning", "so far")) else -1
    return Intent.EVALUATE_DAY.value, {"day_offset_days": day_offset}


def _apply_workout_followup_guard(
    user_text: str,
    intent: str,
    conversation_context: str,
) -> str:
    """Keep scheduled workout nudges in the same coaching thread for alternatives."""
    lowered = user_text.lower()
    followup_signals = (
        "gym closed",
        "gym's closed",
        "can't make it",
        "can't go to the gym",
        "indoor",
        "at home",
        "home workout",
        "alternative",
        "instead",
        "after dinner",
        "different workout",
        "something else",
    )
    if not any(signal in lowered for signal in followup_signals):
        return intent
    ctx = conversation_context.lower()
    scheduled_signals = (
        "workout reminder",
        "coach (scheduled)",
        "on your plan today",
        "full body gym",
    )
    if not any(signal in ctx for signal in scheduled_signals):
        return intent
    if intent == Intent.LOG_NUTRITION.value and is_log_followup_text(user_text):
        return intent
    return Intent.COACHING_CHAT.value


def _apply_nutrition_retry_guard(user_text: str, intent: str) -> str:
    """Avoid UPDATE when the user wants a fresh log after a failed lookup."""
    if intent != Intent.UPDATE_NUTRITION.value:
        return intent
    lowered = user_text.lower()
    retry_signals = (
        "try again",
        "retry",
        "add as a new log",
        "add as new log",
        "new log",
        "log it fresh",
        "log as new",
        "log it again",
        "log again",
        "didn't log",
        "didnt log",
        "wasn't logged",
        "wasnt logged",
        "not logged",
        "never logged",
        "log in the health app",
        "log them",
    )
    correction_signals = (
        "wrong date",
        "wrong time",
        "move to",
        "change to",
        "fix the",
        "update the",
        "correct the",
        "mapped as",
    )
    if any(signal in lowered for signal in correction_signals):
        return intent
    if any(signal in lowered for signal in retry_signals):
        return Intent.LOG_NUTRITION.value
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


def _turn_reset_state() -> dict[str, Any]:
    """Clear terminal fields from a prior graph turn (checkpoint persistence)."""
    return {
        "final_reply": "",
        "pending_confirm": False,
        "conversational_reply": "",
        "intent": "",
        "payload": {},
        "api_result": None,
        "nutrition_search_result": None,
        "research_result": None,
        "vision_result": None,
        "batch_results": [],
        "use_interactive_buttons": False,
    }


def _build_invoke_input(
    *,
    user_text: str,
    sender_phone: str,
    message_type: str,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    image_caption: str = "",
    document_bytes: bytes | None = None,
    document_mime_type: str | None = None,
    document_filename: str = "",
    audio_bytes: bytes | None = None,
    audio_mime_type: str | None = None,
) -> CoachState:
    return {
        **_turn_reset_state(),
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
    }


def _append_nutrition_reply(reply: str, payload: dict[str, Any]) -> str:
    """Ensure WhatsApp replies include resolved macros, not just router preamble."""
    from ..integrations.nutrition import build_nutrition_user_reply

    nutrition_line = (payload.get("nutrition_reply") or "").strip()
    if not nutrition_line and payload.get("calories_kcal") is not None:
        nutrition_line = build_nutrition_user_reply(payload).strip()
    if not nutrition_line or nutrition_line in reply:
        return reply.strip()
    if reply.strip():
        return f"{reply.strip()}\n\n{nutrition_line}".strip()
    return nutrition_line


def _needs_nutrition_confirm(intent: str, payload: dict[str, Any]) -> bool:
    if intent != Intent.LOG_NUTRITION.value:
        return False
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
    vision = VisionAgent(client=engine.vision_llm)

    def _llm_context(state: CoachState, *, include_health_snapshot: bool = True) -> dict[str, str]:
        return build_llm_context(
            sender_phone=state.get("sender_phone", ""),
            user_text=state.get("user_text", "") or state.get("image_caption", ""),
            health_client=client,
            include_health_snapshot=include_health_snapshot,
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

        payload = apply_caption_portion_hints(
            {
                "food_display_name": vision_result.get("food_display_name", "Meal from photo"),
                "portion_description": vision_result.get("portion_description", "1 serving"),
                "meal_type": vision_result.get("meal_type", "MEAL_TYPE_UNSPECIFIED"),
                "vision_notes": vision_result.get("vision_notes", ""),
                "from_image": True,
            },
            caption,
        )
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
        user_text = state.get("user_text", "")
        sender = state.get("sender_phone", "")
        detect_and_store_coaching_focus(user_text)

        if is_log_followup_text(user_text):
            pending = load_pending_nutrition(sender)
            if pending and pending.get("payload"):
                from ..integrations.nutrition import apply_user_stated_macros

                payload = apply_user_stated_macros(
                    dict(pending["payload"]),
                    user_text=user_text,
                    item_context=False,
                )
                return {
                    **_turn_reset_state(),
                    "intent": Intent.LOG_NUTRITION.value,
                    "payload": payload,
                    "conversational_reply": "Got it — logging that meal to Google Health now.",
                }

        if state.get("final_reply"):
            return {}

        ctx = _llm_context(state, include_health_snapshot=False)
        routed = engine.route_message(
            user_text,
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
            coach_state_context=ctx.get("coach_state_context", ""),
        )
        intent, payload = _apply_no_log_guard(user_text, routed.intent.value, routed.payload)
        intent = _apply_plan_context_guard(user_text, intent, ctx["conversation_context"])
        intent = _apply_workout_followup_guard(user_text, intent, ctx["conversation_context"])
        intent = _apply_nutrition_retry_guard(user_text, intent)
        intent = _apply_wellness_plan_phrasing(user_text, intent)
        day_intent, day_payload = _apply_day_review_guard(user_text, intent)
        if day_intent == Intent.EVALUATE_DAY.value:
            return {
                **_turn_reset_state(),
                "intent": day_intent,
                "payload": day_payload,
                "conversational_reply": "",
                "api_result": None,
            }
        return {
            "intent": intent,
            "payload": payload,
            "conversational_reply": routed.conversational_reply,
            "api_result": None,
        }

    def lookup_nutrition(state: CoachState) -> CoachState | Command:
        ctx = _llm_context(state, include_health_snapshot=False)
        payload = dict(state.get("payload", {}))
        item = state.get("batch_item") or payload
        search_result = search_food_nutrition(
            food_display_name=item.get("food_display_name", "meal"),
            portion_description=item.get("portion_description", ""),
            user_message=state["user_text"],
        )
        enriched_payload = engine.resolve_nutrition_macros(
            user_text=state["user_text"],
            payload={**item, "_batch_item": True} if state.get("batch_item") else item,
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
                return {"batch_results": [_batch_result_entry(item=enriched_payload, reply=item_reply, synced=False)]}
            api_result = execute_health_action(
                intent,
                enriched_payload,
                client=client,
                user_text=state.get("user_text", ""),
            )
            return {
                "batch_results": [
                    _batch_result_entry(
                        item=enriched_payload,
                        reply=item_reply,
                        api_result=api_result,
                    )
                ]
            }

        if intent == Intent.QUERY_NUTRITION.value:
            save_pending_nutrition(
                state.get("sender_phone", ""),
                payload=enriched_payload,
                intent=Intent.LOG_NUTRITION.value,
                user_text=state.get("user_text", ""),
            )

        if _needs_nutrition_confirm(intent, enriched_payload):
            save_pending_nutrition(
                state.get("sender_phone", ""),
                payload=enriched_payload,
                intent=intent,
                user_text=state.get("user_text", ""),
            )
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
                clear_pending_nutrition(state.get("sender_phone", ""))
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

    def lookup_exercise_calories(state: CoachState) -> CoachState:
        ctx = _llm_context(state, include_health_snapshot=False)
        payload = dict(state.get("payload", {}))
        from ..services.user_profile import fetch_user_profile_snapshot

        weight_kg = fetch_user_profile_snapshot(client=client).get("weight_kg")
        enriched = resolve_exercise_payload_for_log(
            payload,
            user_text=state["user_text"],
            engine=engine,
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
            weight_kg=float(weight_kg) if weight_kg else None,
        )
        reply = compose_exercise_reply(
            base_reply=state.get("conversational_reply", ""),
            resolved=enriched if not enriched.get("items") else enriched.get("items", [{}])[0],
        )
        return {
            "payload": enriched,
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
            payload={**item, "_batch_item": True},
            search_result=search_result,
            intent=intent,
            conversation_context=ctx["conversation_context"],
            user_profile_context=ctx["user_profile_context"],
        )
        item_reply = compose_nutrition_reply(base_reply="", resolved=enriched)
        if intent == Intent.QUERY_NUTRITION.value or should_skip_health_sync(intent, enriched):
            return {"batch_results": [_batch_result_entry(item=enriched, reply=item_reply, synced=False)]}
        api_result = execute_health_action(
            intent,
            enriched,
            client=client,
            user_text=state.get("user_text", ""),
        )
        return {
            "batch_results": [
                _batch_result_entry(item=enriched, reply=item_reply, api_result=api_result)
            ]
        }

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
                    payload={**item, "_batch_item": True},
                    search_result=search_result,
                    intent=intent,
                    conversation_context=ctx["conversation_context"],
                    user_profile_context=ctx["user_profile_context"],
                )
                item_reply = compose_nutrition_reply(base_reply="", resolved=enriched)
                if intent == Intent.QUERY_NUTRITION.value or should_skip_health_sync(intent, enriched):
                    lines.append(item_reply)
                    batch_results.append(
                        _batch_result_entry(item=enriched, reply=item_reply, synced=False)
                    )
                    continue
                api_result = execute_health_action(
                    intent,
                    enriched,
                    client=client,
                    user_text=state.get("user_text", ""),
                )
                batch_results.append(
                    _batch_result_entry(
                        item=enriched,
                        reply=item_reply,
                        api_result=api_result,
                    )
                )
                if api_result and api_result.get("error"):
                    api_errors.append(
                        f"{enriched.get('food_display_name')}: {api_result.get('message', 'sync failed')}"
                    )
                else:
                    lines.append(item_reply)
        else:
            for entry in batch_results:
                item_reply = entry.get("reply", "")
                sync_error = entry.get("sync_error")
                if sync_error:
                    api_errors.append(
                        f"{entry.get('item', {}).get('food_display_name')}: {sync_error}"
                    )
                elif item_reply:
                    lines.append(item_reply)

        not_logged: list[str] = []
        for entry in batch_results:
            item = entry.get("item") or {}
            item_reply = (entry.get("reply") or "").strip()
            if entry.get("synced") is False or should_skip_health_sync(intent, item):
                label = item.get("food_display_name") or "item"
                if item_reply:
                    not_logged.append(item_reply)
                else:
                    not_logged.append(f"{label}: needs more detail before I can log it")

        reply = "\n\n".join(line for line in lines if line.strip())
        if not_logged:
            reply = "\n\n".join(
                part for part in [reply.strip(), "\n\n".join(not_logged)] if part
            ).strip()
        if api_errors:
            reply += "\n\n(Some items could not sync: " + "; ".join(api_errors) + ")"
        if intent == Intent.LOG_NUTRITION.value and not api_errors:
            try:
                from ..services.coaching import get_daily_health_snapshot
                from ..services.nutrition_plan import format_brief_progress_line

                progress = format_brief_progress_line(get_daily_health_snapshot(client=client))
                if progress:
                    reply = f"{reply}\n\n{progress}".strip()
            except Exception:
                logger.debug("Could not append batch nutrition progress", exc_info=True)
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
        if result and not result.get("error") and state.get("intent") == Intent.LOG_NUTRITION.value:
            clear_pending_nutrition(state.get("sender_phone", ""))
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

    def evaluate_day(state: CoachState) -> CoachState:
        from ..services.coaching import evaluate_day_towards_goals

        payload = state.get("payload", {}) or {}
        day_offset = int(payload.get("day_offset_days", -1))
        try:
            result = evaluate_day_towards_goals(
                day_offset=day_offset,
                user_text=state.get("user_text", ""),
                client=client,
            )
            return {"api_result": result, "final_reply": result.get("message", "")}
        except GoogleAuthRequiredError:
            raise
        except Exception as exc:
            logger.exception("Day review failed: %s", exc)
            return {
                "final_reply": (
                    "I couldn't pull your Google Health logs for that day just now. "
                    "Please try again in a moment."
                ),
                "api_result": {"error": True, "message": str(exc)},
            }

    def finalize_reply(state: CoachState) -> CoachState:
        if state.get("final_reply"):
            return {}
        reply = state.get("conversational_reply", "")
        payload = dict(state.get("payload") or {})
        api_result = state.get("api_result")
        intent_name = state.get("intent", "")

        if api_result and api_result.get("_duplicate_skipped"):
            dup_reply = _append_nutrition_reply(
                str(api_result.get("message") or reply),
                payload,
            )
            return {"final_reply": dup_reply.strip()}

        if api_result and api_result.get("message") and not api_result.get("error"):
            if intent_name in LOCAL_COACH_INTENTS or api_result.get("plan"):
                reply = api_result.get("message", reply)
            elif api_result.get("today_workout") or api_result.get("entries") is not None:
                reply = api_result.get("message", reply)
            elif intent_name == Intent.LOG_EXERCISE.value and api_result.get("logged_count"):
                reply = api_result.get("message", reply)
            elif intent_name == Intent.DELETE_NUTRITION.value:
                reply = api_result.get("message", reply)
            elif intent_name == Intent.UPDATE_NUTRITION.value and api_result.get("message"):
                reply = api_result.get("message", reply)

        if api_result and api_result.get("error"):
            if intent_name in LOCAL_COACH_INTENTS:
                return {"final_reply": api_result.get("message", "I couldn't complete that just now.").strip()}
            write_intents = {
                Intent.LOG_NUTRITION.value,
                Intent.LOG_EXERCISE.value,
                Intent.LOG_HYDRATION.value,
                Intent.LOG_WEIGHT.value,
                Intent.DELETE_NUTRITION.value,
                Intent.UPDATE_NUTRITION.value,
            }
            if intent_name in write_intents:
                reply = (
                    "I couldn't save that to Google Health.\n\n"
                    f"{api_result.get('message', 'Please try again in a moment.')}"
                )
            else:
                reply = (
                    f"{reply}\n\n"
                    f"(Heads up: I couldn't sync with Google Health just now — "
                    f"{api_result.get('message', 'please try again shortly')})"
                )
            return {"final_reply": reply.strip()}

        if api_result and api_result.get("partial_error"):
            if api_result.get("message") and not api_result.get("errors"):
                return {"final_reply": str(api_result["message"]).strip()}
            err_text = "; ".join(api_result.get("errors") or [])
            reply = (
                f"{api_result.get('message', reply)}\n\n"
                f"(Some exercises didn't sync: {err_text})"
            ).strip()
            return {"final_reply": reply}

        if api_result and api_result.get("replacement_strategy"):
            note = api_result.get("message", "")
            if note:
                if intent_name in {
                    Intent.UPDATE_NUTRITION.value,
                    Intent.UPDATE_EXERCISE.value,
                }:
                    reply = note
                else:
                    reply = f"{reply}\n\n{note}".strip()
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
            ctx = _llm_context(state)
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

        if (
            intent_name == Intent.LOG_NUTRITION.value
            and api_result
            and not api_result.get("error")
        ):
            try:
                from ..services.coaching import get_daily_health_snapshot
                from ..services.nutrition_plan import format_brief_progress_line

                snap = get_daily_health_snapshot(client=client)
                progress = format_brief_progress_line(snap)
                if progress and progress not in reply:
                    reply = f"{reply}\n\n{progress}".strip()
            except Exception:
                logger.debug("Could not append nutrition plan progress", exc_info=True)

        if intent_name in {
            Intent.LOG_NUTRITION.value,
            Intent.QUERY_NUTRITION.value,
            Intent.UPDATE_NUTRITION.value,
        }:
            if should_skip_health_sync(intent_name, payload):
                reply = compose_nutrition_reply(base_reply="", resolved=payload)
            else:
                reply = _append_nutrition_reply(reply, payload)

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

    def after_exercise_lookup(state: CoachState) -> str:
        return route_after_exercise_lookup(
            state.get("intent", ""),
            state.get("payload", {}),
        )

    workflow = StateGraph(CoachState)
    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("analyze_food_image", analyze_food_image)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("lookup_nutrition", lookup_nutrition)
    workflow.add_node("lookup_exercise_calories", lookup_exercise_calories)
    workflow.add_node("process_batch_item", process_batch_item)
    workflow.add_node("batch_log_nutrition", batch_log_nutrition)
    workflow.add_node("research_answer", research_answer)
    workflow.add_node("execute_health", execute_health)
    workflow.add_node("query_coach_data", query_coach_data)
    workflow.add_node("build_wellness_plan", build_wellness_plan)
    workflow.add_node("evaluate_day", evaluate_day)
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
            "lookup_exercise_calories": "lookup_exercise_calories",
            "research_answer": "research_answer",
            "execute_health": "execute_health",
            "query_coach_data": "query_coach_data",
            "build_wellness_plan": "build_wellness_plan",
            "evaluate_day": "evaluate_day",
            "finalize_reply": "finalize_reply",
        },
    )
    workflow.add_conditional_edges(
        "route_intent",
        after_intent_route,
        {
            "batch_log_nutrition": "batch_log_nutrition",
            "lookup_nutrition": "lookup_nutrition",
            "lookup_exercise_calories": "lookup_exercise_calories",
            "research_answer": "research_answer",
            "execute_health": "execute_health",
            "query_coach_data": "query_coach_data",
            "build_wellness_plan": "build_wellness_plan",
            "evaluate_day": "evaluate_day",
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
    workflow.add_conditional_edges(
        "lookup_exercise_calories",
        after_exercise_lookup,
        {"execute_health": "execute_health"},
    )
    workflow.add_edge("batch_log_nutrition", END)
    workflow.add_edge("research_answer", END)
    workflow.add_edge("query_coach_data", END)
    workflow.add_edge("build_wellness_plan", END)
    workflow.add_edge("evaluate_day", END)
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
        interrupts = getattr(state, "interrupts", None) or ()
        return bool(interrupts)
    except Exception:
        return False


def _abandon_pending_interrupt(graph, config: dict[str, Any]) -> None:
    """Close a stale confirm interrupt so a new user message starts fresh."""
    try:
        graph.invoke(Command(resume="skip"), config=config)
    except Exception as exc:
        logger.debug("Could not auto-skip pending interrupt: %s", exc)


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
        pending_nutrition = load_pending_nutrition(sender_phone) if is_log_followup_text(user_text) else None
        if pending_nutrition and pending_nutrition.get("payload"):
            logger.info(
                "Using pending nutrition log for %s instead of stale interrupt",
                sender_phone,
            )
            _abandon_pending_interrupt(graph, config)
            result = graph.invoke(
                _build_invoke_input(
                    user_text=user_text,
                    sender_phone=sender_phone,
                    message_type=message_type,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                    image_caption=image_caption,
                    document_bytes=document_bytes,
                    document_mime_type=document_mime_type,
                    document_filename=document_filename,
                    audio_bytes=audio_bytes,
                    audio_mime_type=audio_mime_type,
                ),
                config=config,
            )
        elif _is_confirm_response(user_text) or _is_skip_response(user_text):
            result = graph.invoke(Command(resume=user_text), config=config)
        else:
            logger.info(
                "Abandoning stale nutrition confirm for %s — new message: %r",
                sender_phone,
                user_text[:80],
            )
            _abandon_pending_interrupt(graph, config)
            result = graph.invoke(
                _build_invoke_input(
                    user_text=user_text,
                    sender_phone=sender_phone,
                    message_type=message_type,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                    image_caption=image_caption,
                    document_bytes=document_bytes,
                    document_mime_type=document_mime_type,
                    document_filename=document_filename,
                    audio_bytes=audio_bytes,
                    audio_mime_type=audio_mime_type,
                ),
                config=config,
            )
    else:
        result = graph.invoke(
            _build_invoke_input(
                user_text=user_text,
                sender_phone=sender_phone,
                message_type=message_type,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                image_caption=image_caption,
                document_bytes=document_bytes,
                document_mime_type=document_mime_type,
                document_filename=document_filename,
                audio_bytes=audio_bytes,
                audio_mime_type=audio_mime_type,
            ),
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
