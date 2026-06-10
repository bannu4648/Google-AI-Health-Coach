"""FastAPI routes consumed by the local React dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..core.analytics import health_overview, health_trends, metric_ranges, overview, recent_table, technical_summary

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/overview")
async def get_overview() -> dict:
    return overview()


@router.get("/health/overview")
async def get_health_overview() -> dict:
    return health_overview()


@router.get("/health/trends")
async def get_health_trends(days: int = Query(default=14, ge=1, le=60)) -> dict:
    return health_trends(days=days)


@router.get("/technical/summary")
async def get_technical_summary() -> dict:
    return technical_summary()


@router.get("/events/recent")
async def get_recent_events(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"items": recent_table("events", limit=limit)}


@router.get("/messages")
async def get_messages(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": recent_table("messages", limit=limit)}


@router.get("/llm-calls")
async def get_llm_calls(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": recent_table("llm_calls", limit=limit)}


@router.get("/google-health-calls")
async def get_google_health_calls(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": recent_table("google_health_calls", limit=limit)}


@router.get("/tavily-calls")
async def get_tavily_calls(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": recent_table("tavily_calls", limit=limit)}


@router.get("/health-actions")
async def get_health_actions(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": recent_table("health_actions", limit=limit)}


@router.get("/job-runs")
async def get_job_runs(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"items": recent_table("job_runs", limit=limit)}


@router.get("/coach-notes")
async def get_coach_notes(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": recent_table("coach_notes", limit=limit)}


@router.get("/daily-summary")
async def get_daily_summaries(limit: int = Query(default=30, ge=1, le=365)) -> dict:
    return {"items": recent_table("daily_summaries", limit=limit)}


@router.get("/metrics/ranges")
async def get_metric_ranges() -> dict:
    return metric_ranges()
