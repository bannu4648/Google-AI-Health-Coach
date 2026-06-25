"""
Provider-agnostic multi-agent LLM facade for the health coach bot.

Specialist agents:
- Router (this module) — intent routing and summarization
- Vision (agent/vision.py) — food photo analysis
- Nutrition / research — resolved via Tavily + structured LLM output

Swap providers via LLM_ROUTING_MODE in .env (all_google, gemini_glm, gemini_mistral).
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ..core.types import DATA_TYPE_PROMPT_LIST, normalize_query_payload
from ..core.database import record_llm_call
from ..core.timezone import llm_time_context
from ..integrations.llm import LLMProvider, create_llm_provider
from ..integrations.llm.gemini import RATE_LIMIT_USER_REPLY
from ..integrations.nutrition import search_has_usable_results

load_dotenv()

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    LOG_NUTRITION = "LOG_NUTRITION"
    UPDATE_NUTRITION = "UPDATE_NUTRITION"
    QUERY_NUTRITION = "QUERY_NUTRITION"
    GENERAL_RESEARCH = "GENERAL_RESEARCH"
    LOG_HYDRATION = "LOG_HYDRATION"
    LOG_WEIGHT = "LOG_WEIGHT"
    LOG_EXERCISE = "LOG_EXERCISE"
    UPDATE_EXERCISE = "UPDATE_EXERCISE"
    CREATE_FITNESS_PLAN = "CREATE_FITNESS_PLAN"
    QUERY_FITNESS_PLAN = "QUERY_FITNESS_PLAN"
    COMPLETE_WORKOUT = "COMPLETE_WORKOUT"
    LOG_MOOD = "LOG_MOOD"
    QUERY_MOOD_HISTORY = "QUERY_MOOD_HISTORY"
    LOG_CYCLE = "LOG_CYCLE"
    QUERY_CYCLE = "QUERY_CYCLE"
    QUERY_COACH_DATA = "QUERY_COACH_DATA"
    LOG_GOAL = "LOG_GOAL"
    UPDATE_GOAL = "UPDATE_GOAL"
    QUERY_GOALS = "QUERY_GOALS"
    BUILD_WELLNESS_PLAN = "BUILD_WELLNESS_PLAN"
    SUMMARIZE_DOCUMENT = "SUMMARIZE_DOCUMENT"
    QUERY_HISTORY = "QUERY_HISTORY"
    QUERY_TRENDS = "QUERY_TRENDS"
    QUERY_SLEEP = "QUERY_SLEEP"
    COACHING_CHAT = "COACHING_CHAT"
    EVALUATE_DAY = "EVALUATE_DAY"
    UNDO_LAST_LOG = "UNDO_LAST_LOG"
    DELETE_NUTRITION = "DELETE_NUTRITION"


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


class FitnessPlanResponse(BaseModel):
    week_start_hkt: str = ""
    goals: dict[str, Any] = Field(default_factory=dict)
    weekly_targets: dict[str, Any] = Field(default_factory=dict)
    workouts: list[dict[str, Any]] = Field(default_factory=list)
    conversational_reply: str = ""


class WellnessPlanResponse(BaseModel):
    final_reply: str
    meal_prep_highlights: list[str] = Field(default_factory=list)
    workout_highlights: list[str] = Field(default_factory=list)


class DocumentSummaryResponse(BaseModel):
    final_reply: str
    key_points: list[str] = Field(default_factory=list)


class HealthPayloadFixResponse(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    fix_summary: str = ""


HEALTH_PAYLOAD_FIX_SYSTEM_PROMPT = """You fix Google Health API v4 write payloads after validation errors.

Return JSON with:
- payload: corrected flat payload fields only (same keys the router uses)
- fix_summary: one short sentence describing what you changed

Rules:
- meal_type must be one of: MEAL_TYPE_UNSPECIFIED, BREAKFAST, LUNCH, DINNER, SNACK, BEFORE_BREAKFAST, BEFORE_LUNCH, BEFORE_DINNER, AFTER_DINNER, ANYTIME
- exercise_type must be a valid Exercise.ExerciseType enum value (e.g. RUNNING, STRENGTH_TRAINING, HIIT, WALKING, BIKING, CARDIO_WORKOUT)
- calories_kcal maps to metricsSummary.caloriesKcal on exercise writes — never use activeEnergy
- hydration unit: MILLILITER, CUP_US, or FLUID_OUNCE_US
- Do not remove required fields like food_display_name, display_name, calories_kcal when present
- Only change fields implicated by the API error or clearly invalid
- logged_at_hkt stays as naive HKT YYYY-MM-DDTHH:mm:ss without Z suffix
"""


def _coerce_router_parsed(parsed: Any) -> dict[str, Any]:
    """Normalize router JSON when the model returns a list for batch nutrition."""
    if isinstance(parsed, list):
        items = [item for item in parsed if isinstance(item, dict) and item.get("food_display_name")]
        return {
            "intent": Intent.LOG_NUTRITION.value,
            "payload": {"items": items},
            "conversational_reply": f"Got it — I'll look up and log {len(items)} item(s) for you.",
        }
    if not isinstance(parsed, dict):
        raise ValueError("Router response must be a JSON object or array of food items.")
    return parsed


class SummaryResponse(BaseModel):
    final_reply: str


class ResearchResponse(BaseModel):
    final_reply: str
    source_urls: list[str] = Field(default_factory=list)


class NutritionMacrosResponse(BaseModel):
    resolution: str = Field(
        description=(
            "use_search when Tavily data is reliable; educated_guess when search failed; "
            "ask_followup when too ambiguous; user_stated when the user gave explicit macros"
        )
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

    @field_validator(
        "resolution",
        mode="before",
    )
    @classmethod
    def _normalize_resolution(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @field_validator(
        "source_url",
        "nutrition_source",
        "nutrition_reply",
        "followup_question",
        "notes",
        "sanity_check",
        "confidence",
        "food_display_name",
        mode="before",
    )
    @classmethod
    def _none_to_empty_str(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("source_urls", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return list(value)

    @field_validator("calories_kcal", mode="before")
    @classmethod
    def _coerce_calories_kcal(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(round(float(value)))

    @field_validator("protein_grams", "carbs_grams", "fat_grams", mode="before")
    @classmethod
    def _coerce_macro_float(cls, value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @model_validator(mode="after")
    def validate_resolution_fields(self) -> "NutritionMacrosResponse":
        if self.resolution == "ask_followup":
            return self
        if self.calories_kcal is None:
            raise ValueError("calories_kcal is required unless resolution is ask_followup")
        return self


class ExerciseCaloriesResponse(BaseModel):
    resolution: str = Field(
        description="use_search when Tavily data is reliable; met_estimate when using MET formula with user weight; educated_guess when search failed"
    )
    calories_kcal: int | None = None
    exercise_source: str = ""
    source_url: str = ""
    source_urls: list[str] = Field(default_factory=list)
    confidence: str = "low"
    sanity_check: str = ""
    exercise_reply: str = ""
    notes: str = ""

    @field_validator("resolution", mode="before")
    @classmethod
    def _normalize_resolution(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @field_validator(
        "source_url",
        "exercise_source",
        "exercise_reply",
        "notes",
        "sanity_check",
        "confidence",
        mode="before",
    )
    @classmethod
    def _none_to_empty_str(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("source_urls", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return list(value)

    @field_validator("calories_kcal", mode="before")
    @classmethod
    def _coerce_calories_kcal(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(round(float(value)))

    @model_validator(mode="after")
    def validate_calories(self) -> "ExerciseCaloriesResponse":
        if self.calories_kcal is None:
            raise ValueError("calories_kcal is required")
        return self


class CoachDataSummaryResponse(BaseModel):
    final_reply: str


class CoachDbQueryResponse(BaseModel):
    sql_query: str
    natural_question: str = ""


COACH_DB_SCHEMA = """
TABLE fitness_plans(id, week_start_hkt, goals_json, weekly_targets_json, status, created_at)
TABLE fitness_workouts(id, plan_id, day_of_week, title, exercise_type, duration_minutes, steps_json, completed_at)
  -- day_of_week: 0=Monday .. 6=Sunday
TABLE mood_logs(id, logged_at_hkt, mood_level, notes, tags_json)
TABLE cycle_logs(id, logged_at_hkt, event_type, details_json)
TABLE user_goals(id, category, goal_text, target_json, progress_json, deadline_hkt, status)
TABLE coach_notes(id, category, note, created_at)
TABLE daily_summaries(date_hkt, summary_type, message)
TABLE health_actions(intent, status, created_at)
VIEW v_active_fitness_plan(plan_id, week_start_hkt, day_of_week, title, duration_minutes, steps_json, completed_at)
VIEW v_recent_mood(id, logged_at_hkt, mood_level, notes, tags_json)
VIEW v_active_goals(id, category, goal_text, target_json, progress_json, deadline_hkt, status)

RULES:
- SELECT only. Always use LIMIT.
- Prefer v_active_fitness_plan for workout plan questions.
- For "my plan Tuesday", filter day_of_week (Mon=0, Tue=1, Wed=2, Thu=3).
- Never query: messages, llm_calls, whatsapp_message_dedup, google_health_calls.
- week_start_hkt is YYYY-MM-DD (Monday).

Example queries:
SELECT day_of_week, title, duration_minutes, steps_json FROM v_active_fitness_plan ORDER BY day_of_week LIMIT 20;
SELECT day_of_week, title, steps_json FROM v_active_fitness_plan WHERE day_of_week IN (1, 3) LIMIT 10;
SELECT logged_at_hkt, mood_level, notes FROM mood_logs ORDER BY logged_at_hkt DESC LIMIT 14;
"""


COACH_DB_QUERY_SYSTEM_PROMPT = """You write read-only SQLite SELECT queries for the health coach local database.

Return JSON with exactly:
- sql_query: one SELECT statement against the schema below (must include LIMIT, max 50)
- natural_question: the user's question restated in plain English

Schema:
{coach_db_schema}

Rules:
- SELECT only — never INSERT, UPDATE, DELETE, or DDL
- Single statement, no semicolons
- Only query tables/views listed in the schema
- Prefer v_active_fitness_plan for workout plan questions
- day_of_week: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
- For Tuesday + Thursday gym sessions: WHERE day_of_week IN (1, 3)
- If a previous query failed, fix the SQL using the error hint provided
""".replace("{coach_db_schema}", COACH_DB_SCHEMA)


ROUTER_SYSTEM_PROMPT = """You are an elite personal wellness coach integrated with Google Health API v4.

The user lives in Hong Kong. All natural-language times are in HKT (UTC+8).
Use the current local date/time provided in each user message to resolve words like
today, yesterday, this morning, last night, and dinner.

Return JSON with exactly these keys:
- intent
- payload
- conversational_reply

Valid intents: LOG_NUTRITION, UPDATE_NUTRITION, DELETE_NUTRITION, QUERY_NUTRITION, GENERAL_RESEARCH, LOG_HYDRATION, LOG_WEIGHT,
LOG_EXERCISE, UPDATE_EXERCISE, CREATE_FITNESS_PLAN, QUERY_FITNESS_PLAN, COMPLETE_WORKOUT,
LOG_MOOD, QUERY_MOOD_HISTORY, LOG_CYCLE, QUERY_CYCLE, QUERY_COACH_DATA,
LOG_GOAL, UPDATE_GOAL, QUERY_GOALS, BUILD_WELLNESS_PLAN, SUMMARIZE_DOCUMENT,
QUERY_HISTORY, QUERY_TRENDS, QUERY_SLEEP, COACHING_CHAT, UNDO_LAST_LOG

Use recent conversation context when the user is correcting or following up on a prior log.

SCHEDULED NUDGES — same conversation thread:
Proactive messages labeled "Coach (scheduled)" (workout reminders, morning/evening summaries, weekly recap)
are part of the same WhatsApp thread. When the user replies to those messages — e.g. "the gym is closed",
"give me indoor alternatives", "something I can do at home after dinner" — use COACHING_CHAT and give
practical workout alternatives grounded in their plan and readiness. Do NOT treat these as unrelated new topics
or re-open an old nutrition lookup unless they clearly switch to food logging.

COACH MEMORY in the user message block summarizes local SQLite state (plans, goals, mood).
For "give me my plan" or today's workout -> QUERY_FITNESS_PLAN (deterministic, no SQL).
For ad-hoc coach history questions (mood counts, goal progress, complex plan filters) -> QUERY_COACH_DATA with sql_query.
Never invent workout steps or plan details in conversational_reply for QUERY_FITNESS_PLAN — say a short ack like "Pulling up your plan…"

conversational_reply MUST be a plain WhatsApp-ready string. Never return an object
such as {"message": "..."} or {"response": "..."} inside conversational_reply.

Agent tool — search_nutrition:
The system runs a Tavily web search against trusted nutrition databases
(USDA FoodData Central, Nutritionix, MyFitnessPal, CalorieKing, Healthline, etc.)
for LOG_NUTRITION, UPDATE_NUTRITION, and QUERY_NUTRITION. You must NOT guess or invent calories or macros.

Agent tool — query_coach_db:
For QUERY_COACH_DATA, the system runs a read-only SQLite text2sql lookup against local coach memory
(plans, goals, mood, cycle logs). You route the question; the query_coach_db tool generates and runs
the SELECT. Put natural_question in the payload (required). sql_query is optional — omit it and let
the tool generate SQL. Never invent plan steps or mood entries in conversational_reply; the tool
returns grounded rows that are summarized into the final reply.

Routing:
- Nutrition lookup ONLY (no app logging) -> QUERY_NUTRITION
- Explicit meal logging to Google Health -> LOG_NUTRITION
- Fixing/correcting a recently logged meal time or details -> UPDATE_NUTRITION
- General sourced/current health question -> GENERAL_RESEARCH
- Plain water (non-alcoholic) -> LOG_HYDRATION
- Alcoholic drinks / cocktails / wine / beer with calories -> LOG_NUTRITION (meal_type SNACK), NOT LOG_HYDRATION
- Workouts / gym / runs / sports logged to Google Health -> LOG_EXERCISE
- Fix a recently logged workout -> UPDATE_EXERCISE
- Create or refresh a weekly fitness plan -> CREATE_FITNESS_PLAN
- Ask what workout is planned today / this week / specific days -> QUERY_FITNESS_PLAN
- Mark today's planned workout complete -> COMPLETE_WORKOUT
- Mood / how they feel -> LOG_MOOD
- Past mood patterns -> QUERY_MOOD_HISTORY
- Period / cycle / symptoms -> LOG_CYCLE
- Cycle history / patterns -> QUERY_CYCLE
- Set a long-term goal -> LOG_GOAL
- Update or complete a goal -> UPDATE_GOAL
- List goals -> QUERY_GOALS
- Build a tailored meal + workout plan from recent logs and goals -> BUILD_WELLNESS_PLAN
- Ad-hoc questions about local coach data (plans, goals, mood history) not in Google Health -> QUERY_COACH_DATA
- Medical PDF or document summary -> SUMMARIZE_DOCUMENT
- Weight/scale -> LOG_WEIGHT
- Past logs/history -> QUERY_HISTORY
- Review a specific day (meals + workouts) vs goals -> EVALUATE_DAY with day_offset_days (-1=yesterday, 0=today)
- Weekly trends/averages -> QUERY_TRENDS
- Sleep -> QUERY_SLEEP
- Undo last log -> UNDO_LAST_LOG (payload: optional data_type hint)
- Delete duplicate/wrong meal logs -> DELETE_NUTRITION (see payload below)
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
meal_type (BREAKFAST|LUNCH|DINNER|SNACK|MEAL_TYPE_UNSPECIFIED),
logged_at_hkt (required when user mentions a time — local Hong Kong clock time as
YYYY-MM-DDTHH:mm:ss with NO timezone suffix, e.g. dinner yesterday 10:30pm -> 2026-06-08T22:30:00)

BATCH LOG_NUTRITION — when user lists 2+ foods/drinks in one message, use:
items: array of {food_display_name, portion_description, meal_type?, logged_at_hkt?}
Do NOT use flat single-item keys when multiple distinct foods are listed.
Never promise to log items that are not in items.
Cap at 8 items. Stagger logged_at_hkt across the evening when user gives a time range.

Do NOT include calories_kcal, protein_grams, carbs_grams, or fat_grams for LOG_NUTRITION.
The search_nutrition tool resolves macros from trusted web sources after routing.

UPDATE_NUTRITION payload:
food_display_name (from conversation if omitted),
portion_description (include only if the user changes quantity/food, not for time-only fixes),
logged_at_hkt (corrected local HKT time, required for time corrections),
meal_type (optional — omit if unchanged)

For multiple meal corrections in one message, use items[] with one object per meal
(food_display_name + logged_at_hkt + optional meal_type each). Cap at 8 items.

Do NOT include calories_kcal or macros unless the user explicitly provides corrected numbers.
For time-only corrections, omit portion_description and all macro fields.

DELETE_NUTRITION payload:
match_keywords (array of substrings, e.g. ["pasta", "linguine"] — matches food_display_name),
keep_display_name (which entry to keep, substring match — required unless delete_all_matches is true),
date_hkt (optional YYYY-MM-DD — limit to that calendar day in HKT; default last 3 days),
keep_logged_at_hkt (optional — prefer keeper with this local time if multiple match keep_display_name),
delete_all_matches (optional boolean — delete every match on date_hkt; omit keep_display_name)

Use delete_all_matches when removing a single mistaken/phantom entry.
Use DELETE_NUTRITION when user asks to remove/delete duplicate meals — NOT UPDATE_NUTRITION.
Never append Z to logged_at_hkt.

LOG_HYDRATION payload:
milliliters (convert oz/cups to mL), unit (MILLILITER|CUP_US|FLUID_OUNCE_US), logged_at_hkt (optional)

LOG_WEIGHT payload:
weight_grams (convert lb/kg to grams — e.g. 76 kg = 76000), notes (optional), logged_at_hkt (optional)
When user gives weight in kg, convert to grams. Weekly weigh-ins update Google Health and coach memory.

LOG_EXERCISE payload:
display_name (required), exercise_type (e.g. RUNNING, STRENGTH_TRAINING, HIIT, YOGA, WALKING, BIKING),
duration_minutes (required when mentioned), logged_at_hkt (optional), notes (optional), calories_kcal (optional — Tavily + profile lookup runs if omitted)
For multiple distinct exercises in one message, use items[] with display_name + duration_minutes per exercise; otherwise one combined session with notes listing the workout.
Do NOT include calories_kcal for LOG_EXERCISE — Tavily exercise lookup resolves burn after routing.

UPDATE_EXERCISE payload:
display_name (from conversation), duration_minutes (optional), logged_at_hkt (optional for time fixes), notes (optional)

CREATE_FITNESS_PLAN payload:
goals (object with goal strings), schedule_notes (string), equipment (string), week_start_hkt (optional YYYY-MM-DD Monday)

QUERY_FITNESS_PLAN payload:
scope (optional: "today" | "full_week" — default full_week when user says "give me the plan"),
day_filter (optional — day name or list, e.g. "Tuesday", "Tuesday,Thursday", or day_of_week int 0-6)

When user says "give me the plan" / "the plan" / "full plan" -> scope: full_week (never today-only).
When user says "what's my workout today" -> scope: today.

LOG_GOAL rules:
- Only use LOG_GOAL when the user states a concrete goal to save.
- If they ask "can you help log my goals?" with no goal yet -> COACHING_CHAT (ask what goal), NOT LOG_GOAL.
- Never save the user's question text as the goal itself.

Weight / nutrition plan requests:
- "meal prep plan", "tailored nutrition plan", "build my meal and workout plan", "analyze my meals and suggest",
  "weight loss plan", "help me lose weight with a plan" -> BUILD_WELLNESS_PLAN
- Simple nutrition history without asking for a plan -> QUERY_HISTORY nutrition-log
- BUILD_WELLNESS_PLAN uses both meal and exercise history plus active goals (and fitness plan if present).
- If user says "the plan" right after weight/meal discussion -> BUILD_WELLNESS_PLAN, NOT QUERY_FITNESS_PLAN.
- QUERY_FITNESS_PLAN only when they clearly mean gym/workout/fitness schedule.

BUILD_WELLNESS_PLAN payload:
focus (optional: weight_loss|meal_prep|general), lookback_days (optional integer, default 21)

QUERY_COACH_DATA payload:
natural_question (required — user's question about local coach memory),
sql_query (optional SELECT — omit and query_coach_db tool will generate it)

Examples that should use QUERY_COACH_DATA (not QUERY_FITNESS_PLAN):
- "how many moods did I log in May?"
- "what goals are active right now?"
- "show mood trend last 2 weeks"

Coach DB schema (reference for optional sql_query only — tool generates SQL if omitted):
{coach_db_schema}

LOG_GOAL payload:
category (fitness|nutrition|sleep|weight|habit), goal_text (required), target (optional object), deadline_hkt (optional)

UPDATE_GOAL payload:
goal_id (optional), goal_text (optional), target (optional), progress (optional), status (active|completed|paused)

QUERY_GOALS payload:
status (optional — active|completed|paused), limit (optional integer, default 10)

COMPLETE_WORKOUT payload:
workout_id (optional — omit to complete today's planned workout), log_to_google_health (boolean, default true)

LOG_MOOD payload:
mood_level (1-5 integer), logged_at_hkt (optional), notes (optional), tags (optional string list)

QUERY_MOOD_HISTORY payload:
limit (optional integer, default 14)

LOG_CYCLE payload:
event_type (period_start|period_end|flow|symptom), logged_at_hkt (optional), details (optional object with flow/symptom notes)

QUERY_CYCLE payload:
limit (optional integer, default 30)

SUMMARIZE_DOCUMENT payload:
filename (optional), user_question (optional — what they want explained from the doc)

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
For QUERY_FITNESS_PLAN and QUERY_COACH_DATA, conversational_reply must be a short acknowledgment only — never list workout steps or invent plan details.
Do not quote calorie numbers in conversational_reply for nutrition intents — macros and source links are filled in later.
conversational_reply should be warm, concise, and use HKT when mentioning times.
When the user logs food or asks about eating, reference NUTRITION PLAN targets in coach memory if present.
After meal logs, the system may append today's calorie/protein progress — keep conversational_reply short.
""".replace("{data_types}", DATA_TYPE_PROMPT_LIST).replace("{coach_db_schema}", COACH_DB_SCHEMA)

NUTRITION_RESOLVE_SYSTEM_PROMPT = """You resolve nutrition macros using Tavily web search results.

The user message includes a mode:
- lookup_only (QUERY_NUTRITION): answer with nutrition facts only — do NOT say "logged" or "saved"
- logging (LOG_NUTRITION / UPDATE_NUTRITION): meal will be written to Google Health — you may say "logged"

The search targets trusted nutrition databases (USDA, Nutritionix, MyFitnessPal, CalorieKing, etc.).
Return JSON with exactly these keys:
- resolution ("use_search", "educated_guess", "ask_followup", or "user_stated")
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
   - NEVER use ask_followup when the user explicitly stated calories and/or protein in their message — use user_stated instead.

4. user_stated — the user gave explicit calories and/or macros in their message (e.g. "650 calories", "48g protein"):
   - Use those numbers even if Tavily suggests different values for a generic product.
   - Estimate missing carbs/fat only when needed to complete the log.
   - nutrition_reply should confirm you used their numbers and invite correction.

General rules:
- Prefer USDA and major nutrition databases over blogs or forums.
- source_url and every entry in source_urls MUST be copied exactly from Tavily result URLs (never fabricate).
- nutrition_reply MUST contain at least one full https:// link when resolution is use_search.
- Treat Tavily's synthesized answer as secondary. If the answer conflicts with source snippets,
  trust the source snippets/URLs and explain the sane value in sanity_check.
- nutrition_reply should be warm, concise, and under 600 characters.
"""

EXERCISE_RESOLVE_SYSTEM_PROMPT = """You estimate active calories burned during exercise using Tavily web search results and the USER PROFILE.

Return JSON with exactly these keys:
- resolution ("use_search", "met_estimate", or "educated_guess")
- calories_kcal (integer — active calories for the full session)
- exercise_source (short source label; empty for met_estimate/educated_guess unless a URL backs it)
- source_url (full https URL from Tavily when resolution is use_search)
- source_urls (up to 3 https URLs from Tavily you relied on)
- confidence ("high", "medium", or "low")
- sanity_check (1 short sentence: does this burn make sense for duration, exercise type, and user weight?)
- exercise_reply (WhatsApp-ready note about the calorie estimate; plain text, no markdown; under 400 chars)
- notes (brief assumptions — e.g. moderate intensity; empty if none)

Resolution rules:
1. use_search — Tavily returned usable calorie/MET data for this exercise and duration:
   - Personalize using user weight from USER PROFILE when the source gives per-kg or MET values.
   - exercise_reply MUST include ~kcal and a full https source URL when available.
2. met_estimate — Tavily weak but duration + exercise type + user weight allow a MET-based estimate:
   - Use standard MET values (strength ~5, walking ~3.5, running ~9.8, HIIT ~8).
   - Say honestly it is a formula estimate in exercise_reply.
3. educated_guess — sparse data; give a conservative estimate from exercise type and duration.

General rules:
- Never fabricate URLs — copy exactly from Tavily results.
- Prefer Healthline, Harvard, ACE, ExRx, and calculator sites over random blogs.
- For combined workouts with multiple moves in notes, estimate total session burn, not per rep.
- Round calories_kcal to a sensible integer.
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

COACH_DATA_SUMMARIZE_SYSTEM_PROMPT = """You answer questions about the user's local coach memory (fitness plans, goals, mood logs).

Given the user's question, a draft reply, and SQLite query rows, write a concise WhatsApp message that:
- Grounds every fact in the query rows — never invent workouts, goals, or mood entries
- Uses plain language, no markdown
- Says clearly if the query returned no rows
- For workout steps_json, format as readable numbered steps when present

Return JSON with key final_reply.
"""

WELLNESS_PLAN_SYSTEM_PROMPT = """You create personalized wellness plans for WhatsApp (Hong Kong user).

Given the user's goal, recent meal logs, workout history, active goals, and any existing fitness plan,
write a structured plan under 3500 characters with these sections (plain text):

1) Goal recap (1-2 sentences)
2) What I noticed from your logs (patterns in meals, alcohol, workout frequency — be specific)
3) Mon-Sun meal prep & swaps (breakfast/lunch/dinner ideas, batch prep tips, alcohol reduction swaps)
4) Workouts this week (use existing fitness plan if provided; otherwise suggest 3-4 sessions)
5) Top 3 actions for this week

Rules:
- Ground every observation in the provided data — do not invent meals or workouts they didn't log
- If meal data is sparse, say so and give conservative general guidance
- Be practical for Hong Kong lifestyle (eating out, busy weekdays)
- Return JSON with final_reply (full WhatsApp message), meal_prep_highlights (3 bullets), workout_highlights (3 bullets)
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
- For exercise calorie burn, personalize using the USER PROFILE when available (weight, age, height).
- Use recent conversation context for follow-ups (e.g. "what about 20 reps?" refers to the prior exercise).
- For exercise calorie burn, explain that exact burn still varies by intensity, form, and duration.
"""


def _system_prompt(base: str, user_profile_context: str = "", coach_state_context: str = "") -> str:
    parts = [base]
    if coach_state_context.strip():
        parts.append(coach_state_context.strip())
    if user_profile_context.strip():
        parts.append(user_profile_context.strip())
    return "\n\n".join(parts)


def _user_prompt(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


class AIEngine:
    """Routes natural-language messages to structured intents via the configured LLM."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        provider: str | None = None,
        call_delay_seconds: float | None = None,
        rate_limit_max_retries: int | None = None,
        rate_limit_backoff_seconds: float | None = None,
    ):
        self._llm = llm or create_llm_provider(
            provider,
            api_key=api_key,
            model_name=model_name,
            call_delay_seconds=call_delay_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        )

    @property
    def _model(self) -> str:
        """Model id for llm_calls logging (updates when failover switches provider)."""
        return self._llm.model_name

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def vision_llm(self) -> LLMProvider:
        """Vision/multimodal provider — Gemini in dual-model modes."""
        vision = getattr(self._llm, "vision", None)
        if vision is not None:
            return vision
        return self._llm

    @property
    def text_llm(self) -> LLMProvider:
        """Text/reasoning provider — GLM or Mistral in dual-model modes."""
        text = getattr(self._llm, "text", None)
        if text is not None:
            return text
        return self._llm

    # Backward-compatible alias used by graph.py (vision should use vision_llm).
    @property
    def _client(self) -> LLMProvider:
        return self._llm

    def _is_rate_limit_error(self, exc: BaseException) -> bool:
        checker = getattr(self._llm, "is_rate_limit_error", None)
        if callable(checker):
            return checker(exc)
        return "429" in str(exc).lower() or "rate limit" in str(exc).lower()

    def _rate_limit_reply(self) -> str:
        return self._llm.rate_limit_user_reply

    def route_message(
        self,
        user_text: str,
        *,
        conversation_context: str = "",
        user_profile_context: str = "",
        coach_state_context: str = "",
    ) -> RouterResponse:
        """
        Parse a WhatsApp message into intent, API payload, and coach reply.

        Falls back to COACHING_CHAT if JSON validation fails.
        """
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            coach_state_context,
            f"User message: {user_text.strip()}",
        )

        started = time.perf_counter()
        try:
            parsed = self._llm.generate_json(
                purpose="route_message",
                system_prompt=_system_prompt(
                    ROUTER_SYSTEM_PROMPT,
                    user_profile_context,
                    coach_state_context,
                ),
                user_prompt=prompt,
                temperature=0.2,
            )
            parsed = _coerce_router_parsed(parsed)
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
            logger.exception("Failed to parse Gemini router response: %s", exc)
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
            logger.exception("Gemini routing error: %s", exc)
            reply = (
                self._rate_limit_reply()
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
        conversation_context: str = "",
        user_profile_context: str = "",
    ) -> dict[str, Any]:
        """Fill calories and macros from Tavily search results before Google Health logging."""
        from ..integrations.nutrition import format_tavily_source_links

        usable = search_has_usable_results(search_result)
        mode = "lookup_only" if intent == Intent.QUERY_NUTRITION.value else "logging"
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            f"Mode: {mode}",
            f"User message: {user_text}",
            f"Extracted food fields: {json.dumps(payload, default=str)}",
            f"Tavily search status: {search_result.get('status')}",
            f"Tavily has usable results: {usable}",
            f"Tavily query: {search_result.get('query', '')}",
            f"Tavily answer: {search_result.get('answer') or 'None'}",
            f"Tavily source links:\n{format_tavily_source_links(search_result)}",
            f"Tavily results: {json.dumps(search_result.get('results', []), default=str)[:5000]}",
            f"Search error (if any): {search_result.get('error') or 'None'}",
            "If Tavily has usable results=false, do NOT use resolution use_search. "
            "Choose educated_guess or ask_followup and explain honestly in nutrition_reply. "
            "When use_search, nutrition_reply MUST include the full https URL from Tavily source links.",
        )
        started = time.perf_counter()
        try:
            raw_parsed = self._llm.generate_json(
                purpose="resolve_nutrition_macros",
                system_prompt=_system_prompt(NUTRITION_RESOLVE_SYSTEM_PROMPT, user_profile_context),
                user_prompt=prompt,
                temperature=0.1,
            )
            try:
                parsed = NutritionMacrosResponse.model_validate(raw_parsed)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                record_llm_call(
                    purpose="resolve_nutrition_macros",
                    model=self._model,
                    status="parse_error",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                    response={"raw": raw_parsed},
                    error=str(exc),
                )
                logger.exception("Nutrition macro parse error: %s", exc)
                parsed = None
            if parsed is None:
                lookup_only = intent == Intent.QUERY_NUTRITION.value
                record_llm_call(
                    purpose="resolve_nutrition_macros",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                    response={"raw": raw_parsed} if "raw_parsed" in locals() else {},
                )
                return {
                    **payload,
                    "nutrition_resolution": "ask_followup",
                    "nutrition_lookup_only": lookup_only,
                    "nutrition_reply": (
                        "I couldn't resolve nutrition numbers for that meal just now. "
                        + (
                            "Say 'log it' once you're happy with an estimate, "
                            "or describe the portion size."
                            if lookup_only
                            else "Nothing was logged — try again or share portion details."
                        )
                    ),
                }
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
            from ..integrations.nutrition import apply_user_stated_macros

            merged = apply_user_stated_macros(
                merged,
                user_text=user_text,
                item_context=bool(payload.get("_batch_item")),
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

    def resolve_exercise_calories(
        self,
        *,
        user_text: str,
        payload: dict[str, Any],
        search_result: dict[str, Any],
        conversation_context: str = "",
        user_profile_context: str = "",
        weight_kg: float | None = None,
    ) -> dict[str, Any]:
        """Fill calories_kcal from Tavily search + user profile before exercise logging."""
        from ..integrations.exercise import format_tavily_source_links
        from ..integrations.nutrition import search_has_usable_results
        from ..core.payloads import enrich_exercise_log_payload, estimate_exercise_calories_kcal

        usable = search_has_usable_results(search_result)
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            f"User message: {user_text}",
            f"Extracted exercise fields: {json.dumps(payload, default=str)}",
            f"User weight for MET math (kg): {weight_kg or 'unknown'}",
            f"Tavily search status: {search_result.get('status')}",
            f"Tavily has usable results: {usable}",
            f"Tavily query: {search_result.get('query', '')}",
            f"Tavily answer: {search_result.get('answer') or 'None'}",
            f"Tavily source links:\n{format_tavily_source_links(search_result)}",
            f"Tavily results: {json.dumps(search_result.get('results', []), default=str)[:5000]}",
            f"Search error (if any): {search_result.get('error') or 'None'}",
            "If Tavily has usable results=false, prefer met_estimate over use_search.",
        )
        started = time.perf_counter()
        try:
            parsed = self._llm.generate_structured(
                purpose="resolve_exercise_calories",
                system_prompt=_system_prompt(EXERCISE_RESOLVE_SYSTEM_PROMPT, user_profile_context),
                user_prompt=prompt,
                response_model=ExerciseCaloriesResponse,
                temperature=0.1,
            )
            if parsed is None:
                record_llm_call(
                    purpose="resolve_exercise_calories",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                )
                merged = dict(payload)
                merged["calories_kcal"] = estimate_exercise_calories_kcal(merged, weight_kg=weight_kg)
                merged["exercise_resolution"] = "met_estimate"
                return enrich_exercise_log_payload(merged, weight_kg=weight_kg)
            merged = dict(payload)
            merged.update(
                {
                    "calories_kcal": parsed.calories_kcal,
                    "exercise_resolution": parsed.resolution,
                    "exercise_source": parsed.exercise_source,
                    "exercise_source_url": parsed.source_url,
                    "exercise_source_urls": parsed.source_urls,
                    "exercise_confidence": parsed.confidence,
                    "exercise_sanity_check": parsed.sanity_check,
                    "exercise_reply": parsed.exercise_reply,
                    "exercise_notes": parsed.notes,
                }
            )
            record_llm_call(
                purpose="resolve_exercise_calories",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                response=parsed.model_dump(),
            )
            return enrich_exercise_log_payload(merged, weight_kg=weight_kg)
        except Exception as exc:
            record_llm_call(
                purpose="resolve_exercise_calories",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "payload": payload, "search_result": search_result},
                error=str(exc),
            )
            logger.exception("Exercise calorie resolve error: %s", exc)
            merged = dict(payload)
            merged["calories_kcal"] = estimate_exercise_calories_kcal(merged, weight_kg=weight_kg)
            merged["exercise_resolution"] = "met_estimate"
            return enrich_exercise_log_payload(merged, weight_kg=weight_kg)

    def answer_research_question(
        self,
        *,
        user_text: str,
        draft_reply: str,
        search_result: dict[str, Any],
        conversation_context: str = "",
        user_profile_context: str = "",
    ) -> str:
        """Answer a sourced general wellness question from Tavily results."""
        from ..integrations.research import format_research_source_links

        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            f"User question: {user_text}",
            f"Draft reply: {draft_reply}",
            f"Tavily query: {search_result.get('query', '')}",
            f"Tavily answer: {search_result.get('answer') or 'None'}",
            f"Tavily source links:\n{format_research_source_links(search_result)}",
            f"Tavily results: {json.dumps(search_result.get('results', []), default=str)[:6000]}",
            f"Search error (if any): {search_result.get('error') or 'None'}",
        )
        started = time.perf_counter()
        try:
            parsed = self._llm.generate_structured(
                purpose="answer_research_question",
                system_prompt=_system_prompt(RESEARCH_SYSTEM_PROMPT, user_profile_context),
                user_prompt=prompt,
                response_model=ResearchResponse,
                temperature=0.2,
            )
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
            logger.exception("Gemini research answer error: %s", exc)
            return draft_reply

    def summarize_health_data(
        self,
        *,
        user_text: str,
        draft_reply: str,
        api_result: dict[str, Any],
        conversation_context: str = "",
        user_profile_context: str = "",
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
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            f"User question: {user_text}",
            f"Draft reply: {draft_reply}",
            f"API data: {serialized}",
        )
        started = time.perf_counter()
        try:
            parsed = self._llm.generate_structured(
                purpose="summarize_health_data",
                system_prompt=_system_prompt(SUMMARIZE_SYSTEM_PROMPT, user_profile_context),
                user_prompt=prompt,
                response_model=SummaryResponse,
                temperature=0.3,
            )
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
            logger.exception("Gemini summarize error: %s", exc)
            return draft_reply

    def summarize_coach_data(
        self,
        *,
        user_text: str,
        draft_reply: str,
        query_result: dict[str, Any],
        natural_question: str = "",
        conversation_context: str = "",
        user_profile_context: str = "",
        coach_state_context: str = "",
    ) -> str:
        """Turn coach SQLite rows into a grounded WhatsApp reply."""
        if query_result.get("error"):
            return query_result.get("message", draft_reply)
        rows = query_result.get("rows") or []
        if not rows:
            return "I couldn't find matching data in your coach history for that question."
        serialized = json.dumps(
            {"rows": rows, "row_count": query_result.get("row_count", len(rows))},
            default=str,
        )
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            f"User question: {natural_question or user_text}",
            f"Draft reply: {draft_reply}",
            f"SQLite rows: {serialized}",
        )
        started = time.perf_counter()
        try:
            parsed = self._llm.generate_structured(
                purpose="summarize_coach_data",
                system_prompt=_system_prompt(
                    COACH_DATA_SUMMARIZE_SYSTEM_PROMPT,
                    user_profile_context,
                    coach_state_context,
                ),
                user_prompt=prompt,
                response_model=CoachDataSummaryResponse,
                temperature=0.2,
            )
            if parsed is None:
                record_llm_call(
                    purpose="summarize_coach_data",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "query_result": query_result},
                )
                return draft_reply
            record_llm_call(
                purpose="summarize_coach_data",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "query_result": query_result},
                response=parsed.model_dump(),
            )
            return parsed.final_reply
        except Exception as exc:
            record_llm_call(
                purpose="summarize_coach_data",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "query_result": query_result},
                error=str(exc),
            )
            logger.exception("Coach data summarize error: %s", exc)
            return draft_reply

    def generate_coach_db_query(
        self,
        *,
        user_text: str,
        natural_question: str = "",
        conversation_context: str = "",
        coach_state_context: str = "",
        error_hint: str = "",
        previous_sql: str = "",
    ) -> CoachDbQueryResponse:
        """Generate a guarded read-only SELECT for query_coach_db."""
        question = natural_question or user_text
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            coach_state_context,
            f"User question: {question}",
            f"Previous SQL (if any): {previous_sql or 'none'}",
            f"Previous error (if any): {error_hint or 'none'}",
        )
        started = time.perf_counter()
        try:
            parsed = self._llm.generate_structured(
                purpose="generate_coach_db_query",
                system_prompt=COACH_DB_QUERY_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=CoachDbQueryResponse,
                temperature=0.1,
            )
            if parsed is None or not parsed.sql_query.strip():
                record_llm_call(
                    purpose="generate_coach_db_query",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "natural_question": question},
                )
                return CoachDbQueryResponse(
                    sql_query=(
                        "SELECT day_of_week, title, duration_minutes, steps_json "
                        "FROM v_active_fitness_plan ORDER BY day_of_week LIMIT 20"
                    ),
                    natural_question=question,
                )
            if not parsed.natural_question:
                parsed.natural_question = question
            record_llm_call(
                purpose="generate_coach_db_query",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "natural_question": question},
                response=parsed.model_dump(),
            )
            return parsed
        except Exception as exc:
            record_llm_call(
                purpose="generate_coach_db_query",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "natural_question": question},
                error=str(exc),
            )
            logger.exception("Coach DB query generation error: %s", exc)
            return CoachDbQueryResponse(
                sql_query=(
                    "SELECT logged_at_hkt, mood_level, notes "
                    "FROM mood_logs ORDER BY logged_at_hkt DESC LIMIT 14"
                ),
                natural_question=question,
            )

    def transcribe_audio(
        self,
        *,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
    ) -> str:
        """Transcribe a WhatsApp voice note via the vision-capable LLM provider."""
        transcribe = getattr(self._llm, "transcribe_audio", None)
        if not callable(transcribe):
            return ""
        try:
            return transcribe(audio_bytes=audio_bytes, mime_type=mime_type) or ""
        except Exception as exc:
            logger.exception("Audio transcription error: %s", exc)
            return ""

    def summarize_document(
        self,
        *,
        document_bytes: bytes,
        mime_type: str,
        filename: str = "",
        user_question: str = "",
        conversation_context: str = "",
        user_profile_context: str = "",
    ) -> str:
        """Summarize a medical or health document."""
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            f"Filename: {filename or 'document'}",
            f"User question: {user_question or 'Summarize this document in plain language.'}",
        )
        summarize = getattr(self._llm, "summarize_document", None)
        if callable(summarize):
            try:
                return summarize(
                    document_bytes=document_bytes,
                    mime_type=mime_type,
                    system_prompt=_system_prompt(
                        "You summarize health and medical documents for a WhatsApp coach in Hong Kong. "
                        "Return plain text under 900 chars. Include a brief disclaimer to consult their doctor "
                        "for medical decisions. Never diagnose.",
                        user_profile_context,
                    ),
                    user_prompt=prompt,
                )
            except Exception as exc:
                logger.exception("Document summarize error: %s", exc)
        return (
            "I received your document but couldn't summarize it right now. "
            "Please try again or describe what you'd like to know."
        )

    def generate_fitness_plan(
        self,
        *,
        user_text: str,
        payload: dict[str, Any],
        conversation_context: str = "",
        user_profile_context: str = "",
    ) -> FitnessPlanResponse | None:
        """Create a structured weekly fitness plan."""
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            f"User request: {user_text}",
            f"Payload: {json.dumps(payload, default=str)}",
            "Return workouts for Mon-Sun (day_of_week 0=Monday .. 6=Sunday) with numbered step strings.",
        )
        system = (
            "You create personalized weekly fitness plans for WhatsApp. "
            "Return JSON: week_start_hkt (YYYY-MM-DD Monday), goals (object), weekly_targets (object), "
            "workouts (list of {day_of_week, title, exercise_type, duration_minutes, steps: [string]}), "
            "conversational_reply (warm summary)."
        )
        try:
            parsed = self._llm.generate_structured(
                purpose="generate_fitness_plan",
                system_prompt=_system_prompt(system, user_profile_context),
                user_prompt=prompt,
                response_model=FitnessPlanResponse,
                temperature=0.3,
            )
            return parsed
        except Exception as exc:
            logger.exception("Fitness plan generation error: %s", exc)
            return None

    def generate_wellness_plan(
        self,
        *,
        user_text: str,
        payload: dict[str, Any],
        wellness_context: dict[str, Any],
        conversation_context: str = "",
        user_profile_context: str = "",
        coach_state_context: str = "",
    ) -> WellnessPlanResponse | None:
        """Create a structured meal + workout wellness plan from recent health data."""
        serialized = json.dumps(wellness_context, default=str)
        if len(serialized) > 14000:
            wellness_context = {
                "truncated": True,
                "nutrition_count": wellness_context.get("nutrition", {}).get("count", 0),
                "exercise_count": wellness_context.get("exercise", {}).get("count", 0),
                "goals": wellness_context.get("goals", []),
                "fitness_plan": wellness_context.get("fitness_plan"),
                "preview": serialized[:14000],
            }
            serialized = json.dumps(wellness_context, default=str)
        prompt = _user_prompt(
            llm_time_context(),
            conversation_context,
            coach_state_context,
            f"User request: {user_text}",
            f"Payload focus: {payload.get('focus', 'general')}",
            f"Wellness context: {serialized}",
        )
        started = time.perf_counter()
        try:
            parsed = self._llm.generate_structured(
                purpose="generate_wellness_plan",
                system_prompt=_system_prompt(WELLNESS_PLAN_SYSTEM_PROMPT, user_profile_context),
                user_prompt=prompt,
                response_model=WellnessPlanResponse,
                temperature=0.35,
            )
            if parsed is None:
                record_llm_call(
                    purpose="generate_wellness_plan",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"user_text": user_text, "payload": payload},
                )
                return None
            record_llm_call(
                purpose="generate_wellness_plan",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "payload": payload},
                response=parsed.model_dump(),
            )
            return parsed
        except Exception as exc:
            record_llm_call(
                purpose="generate_wellness_plan",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"user_text": user_text, "payload": payload},
                error=str(exc),
            )
            logger.exception("Wellness plan generation error: %s", exc)
            return None

    def fix_health_payload_from_error(
        self,
        *,
        intent: str,
        payload: dict[str, Any],
        error_message: str,
        user_text: str = "",
    ) -> dict[str, Any] | None:
        """Ask the LLM to correct a flat write payload after a Google Health API validation error."""
        prompt = _user_prompt(
            llm_time_context(),
            f"Intent: {intent}",
            f"User message: {user_text or '(not provided)'}",
            f"Current payload: {json.dumps(payload, default=str)}",
            f"Google Health API error: {error_message}",
        )
        started = time.perf_counter()
        try:
            parsed = self._llm.generate_structured(
                purpose="fix_health_payload",
                system_prompt=HEALTH_PAYLOAD_FIX_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=HealthPayloadFixResponse,
                temperature=0.1,
            )
            if parsed is None or not parsed.payload:
                record_llm_call(
                    purpose="fix_health_payload",
                    model=self._model,
                    status="empty",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt={"intent": intent, "payload": payload, "error": error_message},
                )
                return None
            merged = dict(payload)
            merged.update(parsed.payload)
            record_llm_call(
                purpose="fix_health_payload",
                model=self._model,
                status="success",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"intent": intent, "payload": payload, "error": error_message},
                response=parsed.model_dump(),
            )
            merged["_fix_summary"] = parsed.fix_summary
            return merged
        except Exception as exc:
            record_llm_call(
                purpose="fix_health_payload",
                model=self._model,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt={"intent": intent, "payload": payload, "error": error_message},
                error=str(exc),
            )
            logger.exception("Health payload fix error: %s", exc)
            return None
