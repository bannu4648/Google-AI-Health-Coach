"""Compatibility entrypoint for `uvicorn main:app`."""

from backend.health_coach.app import app

__all__ = ["app"]
