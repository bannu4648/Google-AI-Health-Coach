"""
Mistral-powered intent router and conversational parser for the health coach bot.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import MistralError
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..core.types import DATA_TYPE_PROMPT_LIST, normalize_query_payload
from ..core.database import record_llm_call
from ..core.timezone import llm_time_context
from ..integrations.nutrition import search_has_usable_results

load_dotenv()

logger = logging.getLogger(__name__)

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_CALL_DELAY_SECONDS = float(os.getenv("MISTRAL_CALL_DELAY_SECONDS", "2"))
MISTRAL_RATE_LIMIT_MAX_RETRIES = int(os.getenv("MISTRAL_RATE_LIMIT_MAX_RETRIES", "3"))
MISTRAL_RATE_LIMIT_BACKOFF_SECONDS = float(
    os.getenv("MISTRAL_RATE_LIMIT_BACKOFF_SECONDS", "2")
)

T = TypeVar("T")

RATE_LIMIT_USER_REPLY = (
    "Mistral's API rate limit was hit — I've queued a short pause and will be ready "
    "again in a few seconds. Please resend your message."
)


class Intent(str, Enum):
    LOG_NUTRITION = "LOG_NUTRITION"
    UPDATE_NUTRITION = "UPDATE_NUTRITION"
    QUERY_NUTRITION = "QUERY_NUTRITION"
    GENERAL_RESEARCH = "GENERAL_RESEARCH"
    LOG_HYDRATION = "LOG_HYDRATION"
    LOG_WEIGHT = "LOG_WEIGHT"
    QUERY_HISTORY = "QUERY_HISTORY"
    QUERY_TRENDS = "QUERY_TRENDS"
    QUERY_SLEEP = "QUERY_SLEEP"
    COACHING_CHAT = "COACHING_CHAT"


GOOGLE_HEALTH_QUERY_INTENTS = {
    Intent.QUERY_HISTORY,
    Intent.QUERY_TRENDS,
    Intent.QUERY_SLEEP,
}

NUTRITION_LOOKUP_INTENTS = {
    Intent.LOG_NUTRITION,
    Intent.UPDATE_NUTRITION,
    Intent.QUERY_NUTRITION,
}


class RouterResponse(BaseModel):
    intent: Intent
    payload: dict[str, Any] = Field(default_factory=dict)
    conversational_reply: str

    @model_validator(mode="before")
    @classmethod
    def coerce_conversational_reply(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw

        normalized = dict(raw)
        reply = normalized.get("conversational_reply")
        if reply is None:
            for key in ("reply", "message", "response", "text", "content"):
                if isinstance(normalized.get(key), str):
                    normalized["conversational_reply"] = normalized[key]
                    return normalized

        if isinstance(reply, dict):
            for key in ("message", "response", "reply", "text", "content"):
                if isinstance(reply.get(key), str):
                    normalized["conversational_reply"] = reply[key]
                    return normalized
            normalized["conversational_reply"] = json.dumps(reply, ensure_ascii=False)
        elif isinstance(reply, list):
            normalized["conversational_reply"] = " ".join(str(item) for item in reply)
        elif reply is not None and not isinstance(reply, str):
            normalized["conversational_reply"] = str(reply)
        return normalized


class SummaryResponse(BaseModel):
    final_reply: str


class ResearchResponse(BaseModel):
    final_reply: str
    source_urls: list[str] = Field(default_factory=list)


class NutritionMacrosResponse(BaseModel):
    resolution: str = Field(
        description="use_search when Tavily data is reliable; educated_guess when search failed but you can estimate; ask_followup when portion/food is too ambiguous to log"
    )
    calories_kcal: int | None = None
    protein_grams: float | None = None
    carbs_grams: float | None = None
    fat_grams: float | None = None
    food_display_name: str
    nutrition_source: str = ""
    source_url: str = ""
    source_urls: list[str] = Field(default_factory=list)
    confidence: str = "low"
    sanity_check: str = ""
    nutrition_reply: str = ""
    followup_question: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def validate_resolution_fields(self) -> "NutritionMacrosResponse":
        if self.resolution == "ask_followup":
            return self
        if self.calories_kcal is None:
            raise ValueError("calories_kcal is required unless resolution is ask_followup")
        return self


ROUTER_SYSTEM_PROMPT = """You are an elite personal wellness coach integrated with Google Health API v4.

The user lives in Hong Kong. All natural-language times are in HKT (UTC+8).
Use the current local date/time provided in each user message to resolve words like
today, yesterday, this morning, last night, and dinner.

Return JSON with exactly these keys:
- intent
- payload
- conversational_reply

Valid intents: LOG_NUTRITION, UPDATE_NUTRITION, QUERY_NUTRITION, GENERAL_RESEARCH, LOG_HYDRATION, LOG_WEIGHT, QUERY_HISTORY, QUERY_TRENDS, QUERY_SLEEP, COACHING_CHAT

Use recent conversation context when the user is correcting or following up on a prior log.

conversational_reply MUST be a plain WhatsApp-ready string. Never return an object
such as {"message": "..."} or {"response": "..."} inside conversational_reply.

Agent tool — search_nutrition:
The system runs a Tavily web search against trusted nutrition databases
(USDA FoodData Central, Nutritionix, MyFitnessPal, CalorieKing, Healthline, etc.)
for LOG_NUTRITION, UPDATE_NUTRITION, and QUERY_NUTRITION. You must NOT guess or invent calories or macros.

Routing:
- Nutrition lookup ONLY (no app logging) -> QUERY_NUTRITION
- Explicit meal logging to Google Health -> LOG_NUTRITION
- Fixing/correcting a recently logged meal time or details -> UPDATE_NUTRITION
- General sourced/current health question -> GENERAL_RESEARCH
- Water/drinks -> LOG_HYDRATION
- Weight/scale -> LOG_WEIGHT
- Past logs/history -> QUERY_HISTORY
- Weekly trends/averages -> QUERY_TRENDS
- Sleep -> QUERY_SLEEP
- General chat -> COACHING_CHAT with empty payload object

GLOBAL no-log rule:
If the user says "don't log", "do not log", "don't save", "lookup only", "just curious",
or "don't add to the app", never use LOG_NUTRITION, LOG_HYDRATION, or LOG_WEIGHT.
Use QUERY_NUTRITION for food facts, a QUERY_* intent for their Google Health data,
GENERAL_RESEARCH for sourced general questions, or COACHING_CHAT for unsourced general advice.

GENERAL_RESEARCH payload:
needs_web_search=true, search_query (specific query string), topics (short list of topics).
Use this when the user asks for sources, says "check online", asks current/factual health guidance,
or asks general questions not answered by their Google Health data, e.g. "how much REM sleep is
generally required?" or "how many calories does pickleball burn?".

CRITICAL — LOG_NUTRITION vs QUERY_NUTRITION:
Use QUERY_NUTRITION when the user only wants nutrition facts and has NOT asked to log/save/record/track/add to the app or Google Health.
Examples of QUERY_NUTRITION:
- "how many calories in 2 chapatis?"
- "what's the nutrition value of a medium banana?"
- "protein in 200g chicken breast?"
- "calories in pad thai — just curious"

Use LOG_NUTRITION ONLY when the user clearly wants the meal saved to their health app, e.g.:
- "log 2 chapatis for dinner"
- "add this to my app / health app / google health"
- "record/track/save this meal"
- diary-style completion: "I had 2 chapatis for dinner", "ate eggs and toast for breakfast"
- follow-up after a lookup: "yes log it", "log that in the app"

If the user only asks for nutrition info, use QUERY_NUTRITION even if they mention a food and portion.
If ambiguous and no clear logging intent, prefer QUERY_NUTRITION and offer to log if they want.

QUERY_NUTRITION payload (same flat keys as LOG_NUTRITION, but no logging occurs):
food_display_name (required),
portion_description (when quantity/size/count is mentioned),
meal_type (optional)

LOG_NUTRITION payload (flat keys only):
food_display_name (required),
portion_description (required when the user mentions quantity, size, or count —
e.g. "2 chapatis", "1 bowl", "200g chicken breast", "medium latte"),
meal_type (BREAKFAST|LUNCH|DINNER|SNACK|UNKNOWN),
logged_at_hkt (required when user mentions a time — local Hong Kong clock time as
YYYY-MM-DDTHH:mm:ss with NO timezone suffix, e.g. dinner yesterday 10:30pm -> 2026-06-08T22:30:00)

Do NOT include calories_kcal, protein_grams, carbs_grams, or fat_grams for LOG_NUTRITION.
The search_nutrition tool resolves macros from trusted web sources after routing.

UPDATE_NUTRITION payload:
food_display_name (from conversation if omitted),
portion_description (include only if the user changes quantity/food, not for time-only fixes),
logged_at_hkt (corrected local HKT time, required for time corrections),
meal_type (optional — omit if unchanged)

Do NOT include calories_kcal or macros unless the user explicitly provides corrected numbers.
For time-only corrections, omit portion_description and all macro fields.

IMPORTANT: Do NOT convert HKT to UTC yourself for logged_at_hkt. Python handles conversion.
Never append Z to logged_at_hkt.

LOG_HYDRATION payload:
milliliters (convert oz/cups to mL), unit (MILLILITER|CUP_US|FLUID_OUNCE_US), logged_at_hkt (optional)

LOG_WEIGHT payload:
weight_grams (convert lb/kg to grams), notes (optional), logged_at_hkt (optional)

QUERY_HISTORY / QUERY_SLEEP / QUERY_TRENDS payload:
data_type, start_time, end_time (ISO8601 UTC — convert HKT range boundaries to UTC),
query_method (optional)

CRITICAL data_type rules — use ONLY these exact v4 strings, never Google Fit legacy names:
{data_types}

Examples:
- steps / walking -> "steps"
- workouts / activities / exercises -> "exercise"
- heart rate / bpm -> "heart-rate"
- resting heart rate -> "daily-resting-heart-rate"
- meals / food logs -> "nutrition-log"
- water -> "hydration-log"
- sleep -> "sleep"

NEVER use: ACTIVITY, com.google.*, STEP_COUNT, or any other legacy identifier.

Query routing hints:
- recent activities/workouts -> QUERY_HISTORY, data_type "exercise"
- step counts over days -> QUERY_TRENDS, data_type "steps", query_method "daily_roll_up"
- current/recent heart rate -> QUERY_TRENDS, data_type "heart-rate", query_method "reconcile"
- exercise/workout trends -> QUERY_TRENDS, data_type "exercise", query_method "reconcile"
- meal history -> QUERY_HISTORY, data_type "nutrition-log"

For LOG_NUTRITION, say you'll look up trusted sources and log it to their app.
For QUERY_NUTRITION, say you'll look up trusted sources and share the numbers (not log unless they ask).
Do not quote calorie numbers in conversational_reply for nutrition intents — macros and source links are filled in later.
conversational_reply should be warm, concise, and use HKT when mentioning times.
""".replace("{data_types}", DATA_TYPE_PROMPT_LIST)

NUTRITION_RESOLVE_SYSTEM_PROMPT = """You resolve nutrition macros using Tavily web search results.

The user message includes a mode:
- lookup_only (QUERY_NUTRITION): answer with nutrition facts only — do NOT say "logged" or "saved"
- logging (LOG_NUTRITION / UPDATE_NUTRITION): meal will be written to Google Health — you may say "logged"

The search targets trusted nutrition databases (USDA, Nutritionix, MyFitnessPal, CalorieKing, etc.).
Return JSON with exactly these keys:
- resolution ("use_search", "educated_guess", or "ask_followup")
- calories_kcal (integer, omit/null only when resolution is ask_followup)
- protein_grams (number, omit/null only when resolution is ask_followup)
- carbs_grams (number, omit/null only when resolution is ask_followup)
- fat_grams (number, omit/null only when resolution is ask_followup)
- food_display_name (refined display name for the log)
- nutrition_source (short source label; empty for educated_guess or ask_followup)
- source_url (primary full https URL from Tavily results; required when resolution is use_search)
- source_urls (list of full https URLs from Tavily results you relied on; include at least source_url; up to 3 links)
- confidence ("high", "medium", or "low")
- sanity_check (1 short sentence: does this data make sense for the food and portion? flag anything odd)
- nutrition_reply (WhatsApp-ready message to the user explaining what you did; plain text, no markdown)
- followup_question (required when resolution is ask_followup; empty string otherwise)
- notes (brief assumptions about portion scaling; empty string if none)

Resolution rules:
1. use_search — Tavily returned usable nutrition data for this exact (or very close) food/portion:
   - Validate calories/macros are plausible (sanity_check). If numbers look wrong for the portion, switch to ask_followup or educated_guess and explain in nutrition_reply.
   - Scale macros to the user's stated portion (e.g. 2 chapatis = 2x one serving).
   - nutrition_reply MUST include: kcal + protein/carbs/fat when available, source name, and the full source_url https link.
   - If source_urls has multiple links, include the primary URL in nutrition_reply; add "More: <url>" for a second source if helpful.
   - logging mode example: "Logged ~240 kcal (6g protein, 48g carbs, 2g fat) for 2 chapatis — USDA FoodData Central: https://fdc.nal.usda.gov/..."
   - lookup_only mode example: "About ~240 kcal (6g protein, 48g carbs, 2g fat) for 2 chapatis — USDA FoodData Central: https://fdc.nal.usda.gov/... Say 'log it' if you want this saved to your app."

2. educated_guess — Tavily returned no results, irrelevant results, or clearly unreliable data:
   - Tell the user honestly that trusted sources did not return a reliable match.
   - Provide a conservative educated estimate.
   - logging mode: say it is an estimate and was logged; invite correction.
   - lookup_only mode: say it is an estimate only; do NOT say logged; offer to log if they want.
   - Do not invent source_url or source_urls.

3. ask_followup — portion size, brand, preparation, or food identity is too ambiguous to log accurately:
   - Do NOT provide calories_kcal or macros (leave null).
   - nutrition_reply MUST explain that online lookup was insufficient and ask one clear follow-up question.
   - Put the question in both followup_question and nutrition_reply.

General rules:
- Prefer USDA and major nutrition databases over blogs or forums.
- source_url and every entry in source_urls MUST be copied exactly from Tavily result URLs (never fabricate).
- nutrition_reply MUST contain at least one full https:// link when resolution is use_search.
- Treat Tavily's synthesized answer as secondary. If the answer conflicts with source snippets,
  trust the source snippets/URLs and explain the sane value in sanity_check.
- nutrition_reply should be warm, concise, and under 600 characters.
"""

SUMMARIZE_SYSTEM_PROMPT = """You are a premium wellness coach replying on WhatsApp to a user in Hong Kong.

Given the user's original question, a draft reply, and Google Health API JSON, write a
final concise WhatsApp message (under 800 chars) that:
- Keeps the supportive coaching tone
- Highlights the most relevant numbers or trends from the API data
- Uses plain language, no markdown
- Mentions if data looks empty or sparse
- Displays ALL dates and clock times in HKT (UTC+8)
- Use normalized records/totals/warnings if present. Do not claim data is missing
  if record_count is greater than zero.
- Mention how many records/days/sessions were found when useful.

The API JSON may include parallel fields like startTimeHKT / endTimeHKT — prefer those
for user-facing times. Never present raw UTC timestamps as if they were local time.

Return JSON with key final_reply.
"""

RESEARCH_SYSTEM_PROMPT = """You answer general wellness questions using Tavily web search results.

Return JSON with exactly:
- final_reply: concise WhatsApp-ready answer, under 900 chars
- source_urls: list of source URLs used

Rules:
- Do not say anything was logged or saved.
- Use the search result snippets and URLs as sources.
- Include at least one full https:// source link in final_reply when sources are available.
- If results are weak, say so and answer conservatively from general knowledge.
- For exercise calorie burn, explain that exact burn varies by body weight, intensity, and duration.
"""


class AIEngine:
    """Routes natural-language messages to structured intents via Mistral JSON mode."""

    _last_call_at: float = 0.0
    _call_lock = threading.Lock()

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = MISTRAL_MODEL,
        call_delay_seconds: float | None = None,
        rate_limit_max_retries: int | None = None,
        rate_limit_backoff_seconds: float | None = None,
    ):
        key = api_key or MISTRAL_API_KEY
        if not key:
            raise ValueError("Set MISTRAL_API_KEY in your .env file.")
        self._client = Mistral(api_key=key)
        self._model = model_name
        self._call_delay_seconds = (
            MISTRAL_CALL_DELAY_SECONDS
            if call_delay_seconds is None
            else call_delay_seconds
        )
        self._rate_limit_max_retries = (
            MISTRAL_RATE_LIMIT_MAX_RETRIES
            if rate_limit_max_retries is None
            else rate_limit_max_retries
        )
        self._rate_limit_backoff_seconds = (
            MISTRAL_RATE_LIMIT_BACKOFF_SECONDS
            if rate_limit_backoff_seconds is None
            else rate_limit_backoff_seconds
        )

    @property
    def call_delay_seconds(self) -> float:
        return self._call_delay_seconds

    @staticmethod
    def _is_rate_limit_error(exc: BaseException) -> bool:
        if isinstance(exc, MistralError) and exc.status_code == 429:
            return True
        message = str(exc).lower()
        return "429" in message or "rate limit" in message

    def _wait_for_call_slot(self) -> None:
        """Ensure a minimum gap between Mistral calls (shared across requests)."""
        if self._call_delay_seconds <= 0:
            return
        with self._call_lock:
            elapsed = time.monotonic() - AIEngine._last_call_at
            remaining = self._call_delay_seconds - elapsed
            if remaining > 0:
                logger.info(
                    "Waiting %.1fs before Mistral call (rate spacing).",
                    remaining,
                )
                time.sleep(remaining)

    def _mark_call_completed(self) -> None:
        with self._call_lock:
            AIEngine._last_call_at = time.monotonic()

    def _throttle_after_llm_call(self) -> None:
        """Pause after each Mistral API call to reduce rate-limit pressure."""
        if self._call_delay_seconds <= 0:
            return
        logger.info(
            "Throttling %.1fs after Mistral call.",
            self._call_delay_seconds,
        )
        time.sleep(self._call_delay_seconds)
        self._mark_call_completed()

    def _rate_limited_call(self, purpose: str, call: Callable[[], T]) -> T:
        """Space Mistral calls and retry transient HTTP 429 rate limits."""
        self._wait_for_call_slot()
        last_exc: BaseException | None = None
        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                result = call()
                self._throttle_after_llm_call()
                return result
            except Exception as exc:
                last_exc = exc
                if (
                    self._is_rate_limit_error(exc)
                    and attempt < self._rate_limit_max_retries
                ):
                    wait = self._rate_limit_backoff_seconds * (2**attempt)
                    logger.warning(
                        "Mistral rate limit during %s (attempt %d/%d); retrying in %.1fs.",
                        purpose,
                        attempt + 1,
                        self._rate_limit_max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def route_message(
        self,
        user_text: str,
        *,
        conversation_context: str = "",
    ) -> RouterResponse:
        """
        Parse a WhatsApp message into intent, API payload, and coach reply.

        Falls back to COACHING_CHAT if JSON validation fails.
        """
        prompt_parts = [llm_time_context()]
        if conversation_context.strip():
            prompt_parts.append(conversation_context.strip())
        prompt_parts.append(f"User message: {user_text.strip()}")
        prompt = "\n\n".join(prompt_parts)

        started = time.perf_counter()
        try:
            response = self._rate_limited_call(
                "route_message",
                lambda: self._client.chat.complete(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                ),
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            routed = RouterResponse.model_validate(parsed)
            if routed.payload and routed.intent in GOOGLE_HEALTH_QUERY_INTENTS:
                routed.payload = normalize_query_payload(
                    routed.payload, intent=routed.intent.value
                )
            record_llm_call(
                purpose="route_message",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "conversation_context": conversation_context},
                response={"raw": parsed, "normalized": routed.model_dump()},
            )
            return routed
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            record_llm_call(
                purpose="route_message",
                model=self._model,
                status="parse_error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "conversation_context": conversation_context},
                response={"raw": locals().get("raw"), "parsed": locals().get("parsed")},
                error=str(exc),
            )
            logger.exception("Failed to parse Mistral router response: %s", exc)
            return RouterResponse(
                intent=Intent.COACHING_CHAT,
                payload={},
                conversational_reply=(
                    "I hit a small parsing snag, but I'm still here for you. "
                    "Could you rephrase that? For example: 'log lunch: chicken salad' "
                    "or 'how did I sleep this week?'"
                ),
            )
        except Exception as exc:
            record_llm_call(
                purpose="route_message",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "conversation_context": conversation_context},
                error=str(exc),
            )
            logger.exception("Mistral routing error: %s", exc)
            reply = (
                RATE_LIMIT_USER_REPLY
                if self._is_rate_limit_error(exc)
                else (
                    "My coaching brain is taking a quick breather. "
                    "Try again in a moment — I'm ready when you are."
                )
            )
            return RouterResponse(
                intent=Intent.COACHING_CHAT,
                payload={},
                conversational_reply=reply,
            )

    def resolve_nutrition_macros(
        self,
        *,
        user_text: str,
        payload: dict[str, Any],
        search_result: dict[str, Any],
        intent: str = "LOG_NUTRITION",
    ) -> dict[str, Any]:
        """Fill calories and macros from Tavily search results before Google Health logging."""
        from ..integrations.nutrition import format_tavily_source_links

        usable = search_has_usable_results(search_result)
        mode = "lookup_only" if intent == Intent.QUERY_NUTRITION.value else "logging"
        prompt = (
            f"{llm_time_context()}\n"
            f"Mode: {mode}\n"
            f"User message: {user_text}\n"
            f"Extracted food fields: {json.dumps(payload, default=str)}\n"
            f"Tavily search status: {search_result.get('status')}\n"
            f"Tavily has usable results: {usable}\n"
            f"Tavily query: {search_result.get('query', '')}\n"
            f"Tavily answer: {search_result.get('answer') or 'None'}\n"
            f"Tavily source links:\n{format_tavily_source_links(search_result)}\n"
            f"Tavily results: {json.dumps(search_result.get('results', []), default=str)[:5000]}\n"
            f"Search error (if any): {search_result.get('error') or 'None'}\n"
            "If Tavily has usable results=false, do NOT use resolution use_search. "
            "Choose educated_guess or ask_followup and explain honestly in nutrition_reply. "
            "When use_search, nutrition_reply MUST include the full https URL from Tavily source links."
        )
        started = time.perf_counter()
        try:
            response = self._rate_limited_call(
                "resolve_nutrition_macros",
                lambda: self._client.chat.parse(
                    response_format=NutritionMacrosResponse,
                    model=self._model,
                    messages=[
                        {"role": "system", "content": NUTRITION_RESOLVE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                ),
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                record_llm_call(
                    purpose="resolve_nutrition_macros",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                )
                return payload
            merged = dict(payload)
            merged.update(
                {
                    "food_display_name": parsed.food_display_name or payload.get("food_display_name"),
                    "nutrition_resolution": parsed.resolution,
                    "nutrition_source": parsed.nutrition_source,
                    "nutrition_source_url": parsed.source_url,
                    "nutrition_source_urls": parsed.source_urls,
                    "nutrition_lookup_only": intent == Intent.QUERY_NUTRITION.value,
                    "nutrition_confidence": parsed.confidence,
                    "nutrition_sanity_check": parsed.sanity_check,
                    "nutrition_reply": parsed.nutrition_reply,
                    "nutrition_followup_question": parsed.followup_question,
                    "nutrition_notes": parsed.notes,
                }
            )
            if parsed.resolution != "ask_followup":
                merged.update(
                    {
                        "calories_kcal": parsed.calories_kcal,
                        "protein_grams": parsed.protein_grams,
                        "carbs_grams": parsed.carbs_grams,
                        "fat_grams": parsed.fat_grams,
                    }
                )
            record_llm_call(
                purpose="resolve_nutrition_macros",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                response=parsed.model_dump(),
            )
            return merged
        except Exception as exc:
            record_llm_call(
                purpose="resolve_nutrition_macros",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                error=str(exc),
            )
            logger.exception("Nutrition macro resolve error: %s", exc)
            return payload

    def answer_research_question(
        self,
        *,
        user_text: str,
        draft_reply: str,
        search_result: dict[str, Any],
    ) -> str:
        """Answer a sourced general wellness question from Tavily results."""
        from ..integrations.research import format_research_source_links

        prompt = (
            f"{llm_time_context()}\n"
            f"User question: {user_text}\n"
            f"Draft reply: {draft_reply}\n"
            f"Tavily query: {search_result.get('query', '')}\n"
            f"Tavily answer: {search_result.get('answer') or 'None'}\n"
            f"Tavily source links:\n{format_research_source_links(search_result)}\n"
            f"Tavily results: {json.dumps(search_result.get('results', []), default=str)[:6000]}\n"
            f"Search error (if any): {search_result.get('error') or 'None'}"
        )
        started = time.perf_counter()
        try:
            response = self._rate_limited_call(
                "answer_research_question",
                lambda: self._client.chat.parse(
                    response_format=ResearchResponse,
                    model=self._model,
                    messages=[
                        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                ),
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                record_llm_call(
                    purpose="answer_research_question",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "draft_reply": draft_reply, "search_result": search_result},
                )
                return draft_reply
            record_llm_call(
                purpose="answer_research_question",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "draft_reply": draft_reply, "search_result": search_result},
                response=parsed.model_dump(),
            )
            return parsed.final_reply
        except Exception as exc:
            record_llm_call(
                purpose="answer_research_question",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "draft_reply": draft_reply, "search_result": search_result},
                error=str(exc),
            )
            logger.exception("Mistral research answer error: %s", exc)
            return draft_reply

    def summarize_health_data(
        self,
        *,
        user_text: str,
        draft_reply: str,
        api_result: dict[str, Any],
    ) -> str:
        """Turn raw API JSON into a coach-style WhatsApp reply."""
        serialized = json.dumps(api_result, default=str)
        if len(serialized) > 12000:
            api_result = {
                "data_truncated_for_llm": True,
                "original_serialized_chars": len(serialized),
                "preview": serialized[:12000],
            }
            serialized = json.dumps(api_result, default=str)
        prompt = (
            f"{llm_time_context()}\n"
            f"User question: {user_text}\n"
            f"Draft reply: {draft_reply}\n"
            f"API data: {serialized}"
        )
        started = time.perf_counter()
        try:
            response = self._rate_limited_call(
                "summarize_health_data",
                lambda: self._client.chat.parse(
                    response_format=SummaryResponse,
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                ),
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                record_llm_call(
                    purpose="summarize_health_data",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "draft_reply": draft_reply, "api_result": api_result},
                )
                return draft_reply
            record_llm_call(
                purpose="summarize_health_data",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "draft_reply": draft_reply, "api_result": api_result},
                response=parsed.model_dump(),
            )
            return parsed.final_reply
        except Exception as exc:
            record_llm_call(
                purpose="summarize_health_data",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "draft_reply": draft_reply, "api_result": api_result},
                error=str(exc),
            )
            logger.exception("Mistral summarize error: %s", exc)
            return draft_reply
