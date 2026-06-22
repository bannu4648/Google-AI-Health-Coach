"""
Tavily-powered nutrition lookup for meal logging.

Uses trusted food/nutrition domains and returns structured search results
for the agent to resolve calories and macros before writing to Google Health.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

from ..core.database import record_tavily_call

load_dotenv()

logger = logging.getLogger(__name__)


def _nutrition_search_enabled() -> bool:
    return os.getenv("ENABLE_NUTRITION_SEARCH", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

TRUSTED_NUTRITION_DOMAINS = [
    "fdc.nal.usda.gov",
    "usda.gov",
    "nutritionix.com",
    "myfitnesspal.com",
    "calorieking.com",
    "verywellfit.com",
    "healthline.com",
    "fatsecret.com",
    "food.gov.hk",
    "eatforhealth.gov.au",
]


def build_nutrition_query(
    *,
    food_display_name: str,
    portion_description: str = "",
    user_message: str = "",
) -> str:
    """Build a Tavily query biased toward authoritative nutrition facts."""
    food = food_display_name.strip()
    portion = _normalize_portion_for_search(portion_description, food)
    parts = [
        portion,
        food if food.lower() not in portion.lower() else "",
        "nutrition facts",
        "calories",
        "protein",
        "carbohydrates",
        "fat",
    ]
    return " ".join(part for part in parts if part)


def _normalize_portion_for_search(portion_description: str, food: str) -> str:
    """Convert chatty portion text into a compact search phrase.

    Tavily can return zero results for verbose phrases like
    "2 medium apples (about 182g each)". If a gram-per-item value is present,
    prefer the total grams because nutrition sites often index gram portions.
    """
    portion = portion_description.strip()
    if not portion:
        return food

    count_match = re.search(r"^\s*(\d+(?:\.\d+)?)\b", portion)
    grams_each_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:each|per)\b",
        portion,
        flags=re.IGNORECASE,
    )
    if count_match and grams_each_match:
        total_grams = float(count_match.group(1)) * float(grams_each_match.group(1))
        total_text = str(int(total_grams)) if total_grams.is_integer() else f"{total_grams:.1f}"
        return f"{total_text} grams {food}".strip()

    # Parenthetical approximations tend to make Tavily matching brittle.
    compact = re.sub(r"\([^)]*\)", "", portion)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact or food


def _sanitize_search_response(response: dict[str, Any]) -> dict[str, Any]:
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
        )
    return {
        "query": response.get("query"),
        "answer": response.get("answer"),
        "results": results,
        "response_time": response.get("response_time"),
    }


def _prefer_food_relevant_results(response: dict[str, Any], food_display_name: str) -> dict[str, Any]:
    """Keep search results that mention the food when Tavily returns mixed pages."""
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", food_display_name.lower())
        if len(token) > 2
    ]
    if not tokens:
        return response

    relevant = []
    for item in response.get("results", []):
        haystack = " ".join(
            str(item.get(field) or "").lower()
            for field in ("title", "url", "content")
        )
        if any(token in haystack for token in tokens):
            relevant.append(item)

    if not relevant:
        return response
    return {**response, "results": relevant}


def _tavily_request_payload(*, user_message: str = "") -> dict[str, Any]:
    return {
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        "max_results": 5,
        "include_answer": "basic",
        "include_domains": TRUSTED_NUTRITION_DOMAINS,
        "user_message": user_message[:500] if user_message else "",
    }


def _tavily_search_params(*, include_domains: bool = True) -> dict[str, Any]:
    params: dict[str, Any] = {
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        "max_results": 5,
        "include_answer": "basic",
    }
    if include_domains:
        params["include_domains"] = TRUSTED_NUTRITION_DOMAINS
    return params


def _persist_tavily_call(
    *,
    query: str,
    food_display_name: str,
    portion_description: str,
    user_message: str,
    status: str,
    latency_ms: int | None = None,
    response: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    result_count = len((response or {}).get("results", []))
    record_tavily_call(
        query=query or "(not run)",
        status=status,
        food_display_name=food_display_name,
        portion_description=portion_description or None,
        latency_ms=latency_ms,
        result_count=result_count,
        request=_tavily_request_payload(user_message=user_message),
        response=response or {},
        error=error,
    )


def search_food_nutrition(
    *,
    food_display_name: str,
    portion_description: str = "",
    user_message: str = "",
) -> dict[str, Any]:
    """
    Search trusted nutrition sources via Tavily.

    Returns a dict with status, query, answer, results, and optional error.
    """
    query = build_nutrition_query(
        food_display_name=food_display_name,
        portion_description=portion_description,
        user_message=user_message,
    )

    if not _nutrition_search_enabled():
        result = {
            "status": "disabled",
            "query": query,
            "answer": None,
            "results": [],
            "error": "Nutrition web search is disabled (ENABLE_NUTRITION_SEARCH=false).",
        }
        _persist_tavily_call(
            query=query,
            food_display_name=food_display_name,
            portion_description=portion_description,
            user_message=user_message,
            status="disabled",
            response=result,
            error=result["error"],
        )
        return result

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        result = {
            "status": "missing_api_key",
            "query": query,
            "answer": None,
            "results": [],
            "error": "Set TAVILY_API_KEY in .env to enable trusted nutrition lookup.",
        }
        _persist_tavily_call(
            query=query,
            food_display_name=food_display_name,
            portion_description=portion_description,
            user_message=user_message,
            status="missing_api_key",
            response=result,
            error=result["error"],
        )
        return result

    started = time.perf_counter()
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query, **_tavily_search_params(include_domains=True))
        sanitized = _prefer_food_relevant_results(
            _sanitize_search_response(response),
            food_display_name,
        )
        fallback_used = False
        if not search_has_usable_results({"status": "success", **sanitized}):
            fallback_response = client.search(
                query,
                **_tavily_search_params(include_domains=False),
            )
            fallback_sanitized = _prefer_food_relevant_results(
                _sanitize_search_response(fallback_response),
                food_display_name,
            )
            if search_has_usable_results({"status": "success", **fallback_sanitized}):
                sanitized = fallback_sanitized
                fallback_used = True
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = {
            "status": "success",
            "query": query,
            "fallback_without_domain_filter": fallback_used,
            **sanitized,
        }
        _persist_tavily_call(
            query=query,
            food_display_name=food_display_name,
            portion_description=portion_description,
            user_message=user_message,
            status="success",
            latency_ms=latency_ms,
            response=result,
        )
        return result
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.exception("Tavily nutrition search failed: %s", exc)
        result = {
            "status": "error",
            "query": query,
            "answer": None,
            "results": [],
            "error": str(exc),
        }
        _persist_tavily_call(
            query=query,
            food_display_name=food_display_name,
            portion_description=portion_description,
            user_message=user_message,
            status="error",
            latency_ms=latency_ms,
            response=result,
            error=str(exc),
        )
        return result


def format_tavily_source_links(search_result: dict[str, Any]) -> str:
    """Format Tavily result URLs for the nutrition resolver prompt."""
    lines = []
    for item in search_result.get("results", []):
        url = item.get("url")
        if not url:
            continue
        title = item.get("title") or url
        lines.append(f"- {title}: {url}")
    return "\n".join(lines) if lines else "(no source links returned)"


def search_has_usable_results(search_result: dict[str, Any]) -> bool:
    """Whether Tavily returned content the resolver can work with."""
    if search_result.get("status") != "success":
        return False
    if search_result.get("answer"):
        return True
    return any(item.get("content") for item in search_result.get("results", []))


_CALORIES_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:kcal|calories?|cals)\b",
    flags=re.IGNORECASE,
)
_PROTEIN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*protein\b",
    flags=re.IGNORECASE,
)
_CARBS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*carbs?\b",
    flags=re.IGNORECASE,
)
_FAT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*fat\b",
    flags=re.IGNORECASE,
)


def _macro_float(match: re.Match[str] | None) -> float | None:
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_user_stated_macros(
    user_text: str = "",
    portion_description: str = "",
) -> dict[str, Any]:
    """Parse explicit calories/macros the user gave in chat or router portion text."""
    combined = " ".join(part for part in (user_text, portion_description) if part)
    stated: dict[str, Any] = {}
    calories = _macro_float(_CALORIES_RE.search(combined))
    protein = _macro_float(_PROTEIN_RE.search(combined))
    carbs = _macro_float(_CARBS_RE.search(combined))
    fat = _macro_float(_FAT_RE.search(combined))

    if calories is not None and calories > 0:
        stated["calories_kcal"] = int(round(calories))
    if protein is not None and protein >= 0:
        stated["protein_grams"] = protein
    if carbs is not None and carbs >= 0:
        stated["carbs_grams"] = carbs
    if fat is not None and fat >= 0:
        stated["fat_grams"] = fat
    return stated


def _estimate_missing_macros(stated: dict[str, Any]) -> dict[str, Any]:
    """Fill carbs/fat when the user gave calories and protein only."""
    calories = stated.get("calories_kcal")
    protein = stated.get("protein_grams")
    if calories is None or protein is None:
        return stated
    if stated.get("carbs_grams") is not None and stated.get("fat_grams") is not None:
        return stated
    remaining = float(calories) - float(protein) * 4
    if remaining <= 0:
        return stated
    fat = stated.get("fat_grams")
    if fat is None:
        fat = max(0.0, round(remaining * 0.35 / 9, 1))
        stated["fat_grams"] = fat
    if stated.get("carbs_grams") is None:
        carbs = max(0.0, round((remaining - float(fat) * 9) / 4, 1))
        stated["carbs_grams"] = carbs
    return stated


def _user_stated_is_meal_total(user_text: str) -> bool:
    """True when the user gave calories for the whole meal, not one line item."""
    lowered = user_text.lower()
    if re.search(
        r"combined total|total of \d|in total|altogether|whole meal|overall",
        lowered,
    ):
        return True
    if "each of the item" in lowered or "search individually" in lowered:
        return True
    if re.search(r"around \d+ calories", lowered) and any(
        word in lowered for word in ("breakfast", "combination", "items", "meal")
    ):
        return True
    return False


def apply_user_stated_macros(
    resolved: dict[str, Any],
    *,
    user_text: str = "",
    item_context: bool = False,
) -> dict[str, Any]:
    """Prefer explicit user-provided nutrition numbers over web-search second-guessing."""
    stated = extract_user_stated_macros(
        user_text,
        str(resolved.get("portion_description") or ""),
    )
    if not stated.get("calories_kcal"):
        return resolved
    if item_context and _user_stated_is_meal_total(user_text):
        return resolved

    merged = dict(resolved)
    merged.update(_estimate_missing_macros(stated))
    merged["nutrition_resolution"] = "user_stated"
    merged["nutrition_confidence"] = "high"
    merged["nutrition_sanity_check"] = (
        "Used the calories and macros you provided in your message."
    )
    lookup_only = merged.get("nutrition_lookup_only", False)
    kcal = merged["calories_kcal"]
    protein = merged.get("protein_grams")
    carbs = merged.get("carbs_grams")
    fat = merged.get("fat_grams")
    macro_bits = []
    if protein is not None:
        macro_bits.append(f"{int(round(protein))}g protein")
    if carbs is not None:
        macro_bits.append(f"{int(round(carbs))}g carbs")
    if fat is not None:
        macro_bits.append(f"{int(round(fat))}g fat")
    macro_text = f" ({', '.join(macro_bits)})" if macro_bits else ""
    food = merged.get("food_display_name") or "meal"
    if lookup_only:
        merged["nutrition_reply"] = (
            f"About ~{kcal} kcal{macro_text} for {food} using the numbers you gave. "
            "Say 'log it' if you want this saved to your app."
        )
    else:
        merged["nutrition_reply"] = (
            f"Logged ~{kcal} kcal{macro_text} for {food} using the numbers you provided. "
            "Reply if you'd like me to adjust."
        )
    return merged


def _format_kcal_label(calories: Any) -> str:
    if calories is None:
        return "unknown calories"
    try:
        value = int(float(calories))
    except (TypeError, ValueError):
        return "unknown calories"
    if value <= 0:
        return "unknown calories"
    return f"~{value} kcal"


def build_nutrition_user_reply(resolved: dict[str, Any]) -> str:
    """Fallback WhatsApp nutrition message when the LLM omits nutrition_reply."""
    resolution = resolved.get("nutrition_resolution", "educated_guess")
    lookup_only = resolved.get("nutrition_lookup_only", False)
    calories = resolved.get("calories_kcal")
    source = resolved.get("nutrition_source", "")
    url = resolved.get("nutrition_source_url", "")
    extra_urls = [
        item
        for item in resolved.get("nutrition_source_urls", [])
        if item and item != url
    ]
    confidence = resolved.get("nutrition_confidence", "")
    sanity = resolved.get("nutrition_sanity_check", "")
    followup = resolved.get("nutrition_followup_question", "")
    notes = resolved.get("nutrition_notes", "")
    action = "About" if lookup_only else "Logged"
    kcal_label = _format_kcal_label(calories)

    if resolution == "ask_followup" or calories is None:
        if followup:
            return (
                "I couldn't find reliable nutrition data online for that food. "
                f"{followup}"
            )
        if lookup_only:
            return (
                "I couldn't find reliable nutrition data for that meal right now. "
                "Try again in a moment, or describe the portion and I'll estimate. "
                "Say 'log it' when you want it saved to your app."
            )
        return (
            "I couldn't find reliable nutrition data for that meal right now, "
            "so nothing was logged. Describe the portion or try again shortly."
        )

    if resolution == "user_stated":
        parts = [f"{action} {kcal_label}"]
        protein = resolved.get("protein_grams")
        carbs = resolved.get("carbs_grams")
        fat = resolved.get("fat_grams")
        macro_bits = []
        if protein is not None:
            macro_bits.append(f"{int(round(float(protein)))}g protein")
        if carbs is not None:
            macro_bits.append(f"{int(round(float(carbs)))}g carbs")
        if fat is not None:
            macro_bits.append(f"{int(round(float(fat)))}g fat")
        if macro_bits:
            parts.append(f"({', '.join(macro_bits)})")
        parts.append("using the numbers you provided.")
        if lookup_only:
            parts.append("Say 'log it' if you want this saved to your app.")
        else:
            parts.append("Reply if you'd like me to adjust.")
        return " ".join(parts)

    if resolution == "use_search":
        parts = [f"{action} {kcal_label}"]
        if source:
            parts.append(f"from {source}")
        if url:
            parts.append(url)
        if extra_urls:
            parts.append(f"More: {extra_urls[0]}")
        if sanity:
            parts.append(f"— {sanity}")
        elif confidence:
            parts.append(f"({confidence} confidence)")
        if lookup_only:
            parts.append("Say 'log it' if you want this saved to your app.")
        return " ".join(parts) + "."

    if lookup_only:
        parts = [
            "I couldn't find a solid match in trusted nutrition databases, "
            f"so my educated estimate is {kcal_label}.",
        ]
        if notes:
            parts.append(notes)
        parts.append("Say 'log it' if you want this saved to your app.")
        return " ".join(parts)

    parts = [
        "I couldn't find a solid match in trusted nutrition databases, "
        f"so I logged an educated estimate of {kcal_label}.",
    ]
    if notes:
        parts.append(notes)
    parts.append("Reply if you'd like me to correct it.")
    return " ".join(parts)


def compose_nutrition_reply(
    *,
    base_reply: str,
    resolved: dict[str, Any],
) -> str:
    """Merge the router reply with the nutrition resolution message."""
    nutrition_reply = (resolved.get("nutrition_reply") or "").strip()
    if not nutrition_reply:
        nutrition_reply = build_nutrition_user_reply(resolved)
    resolution = resolved.get("nutrition_resolution")
    if resolution in {"ask_followup", "user_stated"}:
        return nutrition_reply
    if base_reply.strip():
        return f"{base_reply.strip()}\n\n{nutrition_reply}".strip()
    return nutrition_reply


def should_skip_health_sync(intent: str, resolved: dict[str, Any]) -> bool:
    """Whether to skip writing to Google Health after nutrition lookup."""
    if intent == "QUERY_NUTRITION":
        return True
    if resolved.get("nutrition_resolution") == "ask_followup":
        return True
    calories = resolved.get("calories_kcal")
    if calories is None:
        return True
    try:
        if int(float(calories)) <= 0:
            return True
    except (TypeError, ValueError):
        return True
    return False


def needs_nutrition_lookup(intent: str, payload: dict[str, Any]) -> bool:
    """Whether the graph should run Tavily before executing a nutrition action."""
    from ..core.payloads import expand_nutrition_items

    if intent in {"LOG_NUTRITION", "QUERY_NUTRITION"}:
        if payload.get("items"):
            return bool(expand_nutrition_items(payload))
        if payload.get("nutrition_resolution") in {"use_search", "educated_guess"}:
            try:
                if int(payload.get("calories_kcal") or 0) > 0:
                    return False
            except (TypeError, ValueError):
                pass
        return bool(payload.get("food_display_name"))
    if intent == "UPDATE_NUTRITION":
        if not payload.get("food_display_name") and not payload.get("portion_description"):
            return False
        macro_fields = ("calories_kcal", "protein_grams", "carbs_grams", "fat_grams")
        if any(payload.get(field) is not None for field in macro_fields):
            return True
        return bool(payload.get("portion_description"))
    return False
