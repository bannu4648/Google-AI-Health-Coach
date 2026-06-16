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

    if resolution == "ask_followup":
        if followup:
            return (
                "I couldn't find reliable nutrition data online for that food. "
                f"{followup}"
            )
        return (
            "I couldn't find reliable nutrition data online for that food. "
            "Could you confirm the portion size or brand?"
        )

    if resolution == "use_search":
        parts = [f"{action} ~{calories} kcal"]
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
            f"so my educated estimate is ~{calories} kcal.",
        ]
        if notes:
            parts.append(notes)
        parts.append("Say 'log it' if you want this saved to your app.")
        return " ".join(parts)

    parts = [
        "I couldn't find a solid match in trusted nutrition databases, "
        f"so I logged an educated estimate of ~{calories} kcal.",
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
    if base_reply.strip():
        return f"{base_reply.strip()}\n\n{nutrition_reply}".strip()
    return nutrition_reply


def should_skip_health_sync(intent: str, resolved: dict[str, Any]) -> bool:
    """Whether to skip writing to Google Health after nutrition lookup."""
    if intent == "QUERY_NUTRITION":
        return True
    return resolved.get("nutrition_resolution") == "ask_followup"


def needs_nutrition_lookup(intent: str, payload: dict[str, Any]) -> bool:
    """Whether the graph should run Tavily before executing a nutrition action."""
    from ..core.payloads import expand_nutrition_items

    if intent in {"LOG_NUTRITION", "QUERY_NUTRITION"}:
        if payload.get("items"):
            return bool(expand_nutrition_items(payload))
        return bool(payload.get("food_display_name"))
    if intent == "UPDATE_NUTRITION":
        if not payload.get("food_display_name") and not payload.get("portion_description"):
            return False
        macro_fields = ("calories_kcal", "protein_grams", "carbs_grams", "fat_grams")
        if any(payload.get(field) is not None for field in macro_fields):
            return True
        return bool(payload.get("portion_description"))
    return False
