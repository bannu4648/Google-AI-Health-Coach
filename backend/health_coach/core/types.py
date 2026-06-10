"""
Google Health API v4 data type constants and LLM output normalization.

Mistral often hallucinates legacy Google Fit names (e.g. com.google.step_count.delta).
This module maps those aliases to the correct v4 kebab-case data types.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical v4 data types this bot supports.
SUPPORTED_DATA_TYPES: frozenset[str] = frozenset(
    {
        "nutrition-log",
        "hydration-log",
        "weight",
        "sleep",
        "heart-rate",
        "daily-resting-heart-rate",
        "steps",
        "active-zone-minutes",
        "exercise",
    }
)

# Display list for prompts.
DATA_TYPE_PROMPT_LIST = ", ".join(sorted(SUPPORTED_DATA_TYPES))

# Maps noisy LLM / legacy Fit aliases -> v4 kebab-case type.
_DATA_TYPE_ALIASES: dict[str, str] = {
    # Steps
    "step": "steps",
    "steps": "steps",
    "step_count": "steps",
    "step-count": "steps",
    "stepcount": "steps",
    "com.google.step_count.delta": "steps",
    "com.google.step_count.cumulative": "steps",
    # Heart rate
    "heart_rate": "heart-rate",
    "heartrate": "heart-rate",
    "heart-rate": "heart-rate",
    "bpm": "heart-rate",
    "resting_heart_rate": "daily-resting-heart-rate",
    "resting-heart-rate": "daily-resting-heart-rate",
    "daily_resting_heart_rate": "daily-resting-heart-rate",
    # Activity / exercise
    "activity": "exercise",
    "activities": "exercise",
    "activity_log": "exercise",
    "workout": "exercise",
    "workouts": "exercise",
    "exercise": "exercise",
    "exercises": "exercise",
    # Nutrition / hydration
    "nutrition": "nutrition-log",
    "nutrition_log": "nutrition-log",
    "food": "nutrition-log",
    "meal": "nutrition-log",
    "meals": "nutrition-log",
    "hydration": "hydration-log",
    "water": "hydration-log",
    # Other
    "active_zone_minutes": "active-zone-minutes",
    "azm": "active-zone-minutes",
    "body_weight": "weight",
}


def normalize_data_type(raw: str | None, *, fallback: str = "steps") -> str:
    """
    Convert an LLM-produced data type string to a supported v4 kebab-case type.

    Handles legacy Google Fit identifiers, uppercase enums, and loose synonyms.
    """
    if not raw:
        return fallback

    cleaned = raw.strip()
    lowered = cleaned.lower().replace(" ", "-").replace("_", "-")

    # Direct kebab-case hit.
    if lowered in SUPPORTED_DATA_TYPES:
        return lowered

    # Alias table (normalize separators first).
    alias_key = cleaned.lower().replace(" ", "_").replace("-", "_")
    if alias_key in _DATA_TYPE_ALIASES:
        return _DATA_TYPE_ALIASES[alias_key]

    compact = re.sub(r"[^a-z0-9]", "", lowered)
    for alias, canonical in _DATA_TYPE_ALIASES.items():
        if re.sub(r"[^a-z0-9]", "", alias) == compact:
            return canonical

    # Legacy com.google.* identifiers.
    if "step" in lowered:
        return "steps"
    if "heart" in lowered and "rest" in lowered:
        return "daily-resting-heart-rate"
    if "heart" in lowered:
        return "heart-rate"
    if "sleep" in lowered:
        return "sleep"
    if "nutrition" in lowered or "food" in lowered or "meal" in lowered:
        return "nutrition-log"
    if "hydration" in lowered or "water" in lowered:
        return "hydration-log"
    if "weight" in lowered:
        return "weight"
    if "exercise" in lowered or "activ" in lowered or "workout" in lowered:
        return "exercise"

    return fallback


def normalize_query_payload(payload: dict[str, Any], *, intent: str) -> dict[str, Any]:
    """Normalize data_type and query_method on query payloads before API calls."""
    if intent == "QUERY_NUTRITION":
        return dict(payload)

    normalized = dict(payload)

    if intent == "QUERY_SLEEP":
        normalized["data_type"] = "sleep"
    elif "data_type" in normalized or intent.startswith("QUERY_"):
        fallback = "steps" if intent == "QUERY_TRENDS" else "nutrition-log"
        if intent == "QUERY_SLEEP":
            fallback = "sleep"
        normalized["data_type"] = normalize_data_type(
            normalized.get("data_type"), fallback=fallback
        )

    method = normalized.get("query_method", "")
    if intent == "QUERY_TRENDS" and not method:
        # Bucketed summaries for rollup-capable interval/sample types.
        if normalized.get("data_type") in {"steps", "active-zone-minutes"}:
            normalized["query_method"] = "daily_roll_up"
        elif normalized.get("data_type") in {"heart-rate", "exercise"}:
            normalized["query_method"] = "reconcile"
        else:
            normalized["query_method"] = "daily_roll_up"
    elif intent == "QUERY_HISTORY" and not method:
        normalized["query_method"] = "list"
    elif intent == "QUERY_SLEEP" and not method:
        normalized["query_method"] = "reconcile"

    return normalized
