"""
Vision agent — analyzes WhatsApp food photos before nutrition lookup/logging.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..core.timezone import llm_time_context
from ..agent.engine import _system_prompt
from ..integrations.llm import LLMProvider
from ..integrations.llm.factory import create_llm_provider

logger = logging.getLogger(__name__)

_LOG_CAPTION_PHRASES = (
    "log it",
    "log this",
    "log that",
    "save it",
    "save this",
    "record it",
    "record this",
    "add to app",
    "add to my app",
    "add to google health",
    "log to",
    "track this",
    "track it",
)


def _caption_requests_logging(caption: str) -> bool:
    lowered = caption.lower()
    return any(phrase in lowered for phrase in _LOG_CAPTION_PHRASES)


VISION_SYSTEM_PROMPT = """You are a food-vision specialist for a personal health coach in Hong Kong.

Analyze the meal photo and return JSON with exactly these keys:
- food_display_name (string, concise label e.g. "Chicken rice with vegetables")
- portion_description (string, best estimate e.g. "1 plate (~350g)")
- meal_type (BREAKFAST|LUNCH|DINNER|SNACK|MEAL_TYPE_UNSPECIFIED)
- wants_to_log (boolean) — true if the user caption clearly asks to log/save/record/add to app
- lookup_only (boolean) — true if caption says don't log, just curious, or no logging intent
- confidence (high|medium|low)
- vision_notes (short assumptions about hidden sauces/oils/portion)
- conversational_reply (warm WhatsApp-ready string; say what you see; do NOT quote calories)

Rules:
- Default for meal photos is lookup_only=true and wants_to_log=false — share nutrition info only, do NOT log.
- If caption is empty, keep lookup_only=true and wants_to_log=false.
- Set wants_to_log=true and lookup_only=false ONLY when the caption clearly asks to log/save/record/track/add to app/Google Health (e.g. "log this", "save to my app").
- If caption says "don't log" or "just curious", set lookup_only=true and wants_to_log=false.
- conversational_reply should describe what you see and say you'll look up nutrition facts (not log unless they asked).
- Use HKT context from the prompt for meal_type timing hints only.
- Never invent exact calories or macros — those come from a later nutrition agent.
"""


class VisionAnalysis(BaseModel):
    food_display_name: str
    portion_description: str = ""
    meal_type: str = "MEAL_TYPE_UNSPECIFIED"
    wants_to_log: bool = False
    lookup_only: bool = True
    confidence: str = "medium"
    vision_notes: str = ""
    conversational_reply: str = ""


class VisionAgent:
    """Vision specialist for meal photos (requires a vision-capable provider, e.g. Gemini)."""

    def __init__(self, client: LLMProvider | None = None, *, vision_client: LLMProvider | None = None):
        self._client = vision_client or client or create_llm_provider()

    def analyze_food_image(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        caption: str = "",
        conversation_context: str = "",
        user_profile_context: str = "",
    ) -> dict[str, Any]:
        prompt_parts = [llm_time_context()]
        if conversation_context.strip():
            prompt_parts.append(conversation_context.strip())
        prompt_parts.append(
            f"User caption: {caption.strip() or '(no caption — nutrition lookup only, do not log)'}"
        )
        user_prompt = "\n\n".join(prompt_parts)

        try:
            parsed = self._client.generate_structured(
                purpose="analyze_food_image",
                system_prompt=_system_prompt(VISION_SYSTEM_PROMPT, user_profile_context),
                user_prompt=user_prompt,
                response_model=VisionAnalysis,
                temperature=0.2,
                images=[(image_bytes, mime_type)],
            )
        except Exception as exc:
            logger.exception("Vision agent %s error: %s", self._client.provider_name, exc)
            parsed = None

        if parsed is None:
            return {
                "food_display_name": "Meal from photo",
                "portion_description": "1 serving",
                "meal_type": "MEAL_TYPE_UNSPECIFIED",
                "wants_to_log": _caption_requests_logging(caption),
                "lookup_only": not _caption_requests_logging(caption),
                "confidence": "low",
                "vision_notes": "Vision parse failed; using generic meal label.",
                "conversational_reply": (
                    "I can see your meal photo, but I'm not fully confident what it is. "
                    "I'll do my best — tell me the dish name or portion if this looks off."
                ),
            }
        return parsed.model_dump()
