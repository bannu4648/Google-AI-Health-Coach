"""Tavily search for general health and wellness research questions."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from ..core.database import record_tavily_call

load_dotenv()

logger = logging.getLogger(__name__)

TRUSTED_HEALTH_DOMAINS = [
    "nih.gov",
    "cdc.gov",
    "who.int",
    "mayoclinic.org",
    "clevelandclinic.org",
    "sleepfoundation.org",
    "healthline.com",
    "nhs.uk",
    "acefitness.org",
]


def _sanitize_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": response.get("query"),
        "answer": response.get("answer"),
        "response_time": response.get("response_time"),
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
            for item in response.get("results", [])
        ],
    }


def format_research_source_links(search_result: dict[str, Any]) -> str:
    lines = []
    for item in search_result.get("results", []):
        url = item.get("url")
        if not url:
            continue
        lines.append(f"- {item.get('title') or url}: {url}")
    return "\n".join(lines) if lines else "(no source links returned)"


def search_health_topic(query: str, *, user_message: str = "") -> dict[str, Any]:
    """Search trusted health sources and persist the Tavily request/response."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    request_payload = {
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        "max_results": 5,
        "include_answer": "basic",
        "include_domains": TRUSTED_HEALTH_DOMAINS,
        "user_message": user_message[:500],
    }
    if not api_key:
        result = {
            "status": "missing_api_key",
            "query": query,
            "answer": None,
            "results": [],
            "error": "Set TAVILY_API_KEY in .env to enable health research search.",
        }
        record_tavily_call(
            query=query,
            status="missing_api_key",
            request=request_payload,
            response=result,
            error=result["error"],
        )
        return result

    started = time.perf_counter()
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query,
            search_depth=request_payload["search_depth"],
            max_results=request_payload["max_results"],
            include_answer=request_payload["include_answer"],
            include_domains=TRUSTED_HEALTH_DOMAINS,
        )
        result = {"status": "success", **_sanitize_response(response)}
        record_tavily_call(
            query=query,
            status="success",
            latency_ms=int((time.perf_counter() - started) * 1000),
            result_count=len(result.get("results", [])),
            request=request_payload,
            response=result,
        )
        return result
    except Exception as exc:
        logger.exception("Tavily health research search failed: %s", exc)
        result = {
            "status": "error",
            "query": query,
            "answer": None,
            "results": [],
            "error": str(exc),
        }
        record_tavily_call(
            query=query,
            status="error",
            latency_ms=int((time.perf_counter() - started) * 1000),
            request=request_payload,
            response=result,
            error=str(exc),
        )
        return result
