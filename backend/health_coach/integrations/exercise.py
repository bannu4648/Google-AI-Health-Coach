"""
Tavily-powered exercise calorie lookup before workout logging.

Mirrors the nutrition search flow: trusted fitness sources → LLM resolution → Google Health write.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from ..core.database import record_tavily_call
from ..core.payloads import expand_exercise_items
from ..integrations.nutrition import format_tavily_source_links, search_has_usable_results

load_dotenv()

logger = logging.getLogger(__name__)

TRUSTED_EXERCISE_DOMAINS = [
    "healthline.com",
    "verywellfit.com",
    "harvard.edu",
    "acefitness.org",
    "calculator.net",
    "exrx.net",
    "livestrong.com",
    "myfitnesspal.com",
    "calorieking.com",
]


def _exercise_search_enabled() -> bool:
    return os.getenv("ENABLE_EXERCISE_SEARCH", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_exercise_calorie_query(
    *,
    display_name: str,
    exercise_type: str = "",
    duration_minutes: int | float | None = None,
    notes: str = "",
    weight_kg: float | None = None,
) -> str:
    """Build a Tavily query for calories burned during an exercise session."""
    parts = []
    if duration_minutes:
        parts.append(f"{int(duration_minutes)} minutes")
    name = (display_name or exercise_type or "workout").strip()
    if name:
        parts.append(name)
    if notes.strip():
        parts.append(notes.strip()[:120])
    if weight_kg:
        parts.append(f"{weight_kg:g} kg person")
    parts.extend(["calories burned", "MET", "energy expenditure"])
    return " ".join(part for part in parts if part)


def _tavily_search_params(*, include_domains: bool = True) -> dict[str, Any]:
    params: dict[str, Any] = {
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        "max_results": 5,
        "include_answer": "basic",
    }
    if include_domains:
        params["include_domains"] = TRUSTED_EXERCISE_DOMAINS
    return params


def _persist_tavily_call(
    *,
    query: str,
    display_name: str,
    status: str,
    user_message: str = "",
    latency_ms: int | None = None,
    response: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    record_tavily_call(
        query=query or "(not run)",
        status=status,
        food_display_name=display_name,
        portion_description="exercise_calories",
        latency_ms=latency_ms,
        result_count=len((response or {}).get("results", [])),
        request={
            "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            "max_results": 5,
            "include_answer": "basic",
            "context": "exercise_calories",
            "user_message": user_message[:500] if user_message else "",
        },
        response=response or {},
        error=error,
    )


def search_exercise_calories(
    *,
    display_name: str,
    exercise_type: str = "",
    duration_minutes: int | float | None = None,
    notes: str = "",
    weight_kg: float | None = None,
    user_message: str = "",
) -> dict[str, Any]:
    """Search trusted fitness sources for calories burned."""
    query = build_exercise_calorie_query(
        display_name=display_name,
        exercise_type=exercise_type,
        duration_minutes=duration_minutes,
        notes=notes,
        weight_kg=weight_kg,
    )

    if not _exercise_search_enabled():
        result = {
            "status": "disabled",
            "query": query,
            "answer": None,
            "results": [],
            "error": "Exercise web search is disabled (ENABLE_EXERCISE_SEARCH=false).",
        }
        _persist_tavily_call(
            query=query,
            display_name=display_name,
            status="disabled",
            user_message=user_message,
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
            "error": "Set TAVILY_API_KEY in .env to enable exercise calorie lookup.",
        }
        _persist_tavily_call(
            query=query,
            display_name=display_name,
            status="missing_api_key",
            user_message=user_message,
            response=result,
            error=result["error"],
        )
        return result

    started = time.perf_counter()
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query, **_tavily_search_params(include_domains=True))
        sanitized = _sanitize_search_response(response)
        fallback_used = False
        if not search_has_usable_results({"status": "success", **sanitized}):
            fallback_response = client.search(query, **_tavily_search_params(include_domains=False))
            fallback_sanitized = _sanitize_search_response(fallback_response)
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
            display_name=display_name,
            status="success",
            user_message=user_message,
            latency_ms=latency_ms,
            response=result,
        )
        return result
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.exception("Tavily exercise search failed: %s", exc)
        result = {
            "status": "error",
            "query": query,
            "answer": None,
            "results": [],
            "error": str(exc),
        }
        _persist_tavily_call(
            query=query,
            display_name=display_name,
            status="error",
            user_message=user_message,
            latency_ms=latency_ms,
            response=result,
            error=str(exc),
        )
        return result


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


def needs_exercise_calorie_lookup(intent: str, payload: dict[str, Any]) -> bool:
    """Whether to run Tavily + LLM before logging exercise."""
    if intent not in {"LOG_EXERCISE", "UPDATE_EXERCISE"}:
        return False
    items = expand_exercise_items(payload)
    if not items:
        return False
    return any(_item_needs_calorie_lookup(item) for item in items)


def _item_needs_calorie_lookup(item: dict[str, Any]) -> bool:
    try:
        calories = int(item.get("calories_kcal") or 0)
    except (TypeError, ValueError):
        calories = 0
    return calories <= 0


def compose_exercise_reply(*, base_reply: str, resolved: dict[str, Any]) -> str:
    """Append calorie resolution notes to the router reply."""
    exercise_reply = (resolved.get("exercise_reply") or "").strip()
    if not exercise_reply:
        kcal = resolved.get("calories_kcal")
        if kcal:
            source = resolved.get("exercise_source") or ""
            url = resolved.get("exercise_source_url") or ""
            exercise_reply = f"Estimated ~{kcal} kcal active burn"
            if source:
                exercise_reply += f" ({source})"
            if url:
                exercise_reply += f" — {url}"
    if base_reply.strip() and exercise_reply:
        return f"{base_reply.strip()}\n\n{exercise_reply}".strip()
    return exercise_reply or base_reply.strip()
