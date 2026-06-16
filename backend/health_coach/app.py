"""
FastAPI application entrypoint.

Mounts the dashboard API and WhatsApp webhook routes. Use with:

    uvicorn backend.health_coach.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.dashboard import router as dashboard_router
from .api.google_oauth import router as google_oauth_router
from .api.webhook import router as webhook_router
from .services.scheduler import start_scheduler, stop_scheduler

load_dotenv()

logging.basicConfig(level=logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()

app = FastAPI(
    title="WhatsApp AI Health Coach",
    version="1.0.0",
    description="Local WhatsApp health coach backed by Mistral, Google Health API v4, and Tavily.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(dashboard_router)
app.include_router(google_oauth_router)
app.include_router(webhook_router)

if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="dashboard-assets",
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def dashboard_index() -> FileResponse:
    """Serve the built dashboard when `frontend/dist` exists."""
    return FileResponse(FRONTEND_DIST / "index.html")
