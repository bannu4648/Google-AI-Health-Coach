"""
User profile context for LLM prompts — Google Health profile + latest body metrics.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv

from ..integrations.google_auth import GoogleAuthRequiredError
from ..integrations.google_health import GoogleHealthAPIError, GoogleHealthClient

load_dotenv()

logger = logging.getLogger(__name__)

PROFILE_CACHE_SECONDS = int(os.getenv("USER_PROFILE_CACHE_SECONDS", "3600"))
MAX_COACH_REPLY_CHARS = int(os.getenv("CONVERSATION_MAX_REPLY_CHARS", "600"))

_cache: dict[str, Any] = {"expires_at": 0.0, "snapshot": {}}


def _env_override(key: str) -> str:
    return os.getenv(key, "").strip()


def _parse_height_meters(height_block: dict[str, Any]) -> float | None:
    if not height_block:
        return None
    for key in ("heightMeters", "inMeters", "meters"):
        value = height_block.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    for key in ("heightMillimeters", "heightMm", "millimeters"):
        value = height_block.get(key)
        if value is not None:
            try:
                return float(value) / 1000.0
            except (TypeError, ValueError):
                pass
    return None


def _parse_weight_kg(weight_block: dict[str, Any]) -> float | None:
    if not weight_block:
        return None
    grams = weight_block.get("weightGrams")
    if grams is not None:
        try:
            return float(grams) / 1000.0
        except (TypeError, ValueError):
            return None
    return None


def _latest_sample_point(
    client: GoogleHealthClient,
    data_type: str,
    *,
    lookback_days: int = 730,
) -> dict[str, Any] | None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    try:
        result = client.reconcile_data_points(
            data_type,
            start_time=start,
            end_time=end,
            page_size=10,
        )
    except (GoogleHealthAPIError, ValueError) as exc:
        logger.info("Could not fetch latest %s sample: %s", data_type, exc)
        return None

    points = result.get("dataPoints", [])
    if not points:
        return None

    def _sample_key(point: dict[str, Any]) -> str:
        block = point.get(data_type, {})
        if not isinstance(block, dict):
            return ""
        sample = block.get("sampleTime", {})
        if not isinstance(sample, dict):
            return ""
        return str(sample.get("physicalTime") or "")

    points.sort(key=_sample_key, reverse=True)
    return points[0]


def fetch_user_profile_snapshot(
    *,
    client: GoogleHealthClient | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load age, height, weight, and stride data from Google Health (+ optional .env overrides)."""
    global _cache
    now = time.monotonic()
    if not force_refresh and _cache["snapshot"] and now < _cache["expires_at"]:
        return dict(_cache["snapshot"])

    health = client or GoogleHealthClient()
    snapshot: dict[str, Any] = {
        "age_years": None,
        "height_cm": None,
        "weight_kg": None,
        "walking_stride_mm": None,
        "running_stride_mm": None,
        "membership_start_date": None,
        "sources": [],
    }

    try:
        profile = health.get_profile()
        if profile.get("age") is not None:
            snapshot["age_years"] = int(profile["age"])
            snapshot["sources"].append("google_health_profile")
        if profile.get("membershipStartDate"):
            snapshot["membership_start_date"] = profile["membershipStartDate"]
        for field, key in (
            ("userConfiguredWalkingStrideLengthMm", "walking_stride_mm"),
            ("autoWalkingStrideLengthMm", "walking_stride_mm"),
            ("userConfiguredRunningStrideLengthMm", "running_stride_mm"),
            ("autoRunningStrideLengthMm", "running_stride_mm"),
        ):
            value = profile.get(field)
            if value is not None and snapshot[key] is None:
                snapshot[key] = int(value)
    except (GoogleHealthAPIError, GoogleAuthRequiredError, ValueError, TypeError) as exc:
        logger.info("Google Health profile unavailable: %s", exc)
        if isinstance(exc, GoogleAuthRequiredError):
            snapshot["auth_required"] = True

    try:
        weight_point = _latest_sample_point(health, "weight")
        if weight_point:
            kg = _parse_weight_kg(weight_point.get("weight", {}))
            if kg is not None:
                snapshot["weight_kg"] = round(kg, 2)
                snapshot["sources"].append("google_health_weight")

        height_point = _latest_sample_point(health, "height")
        if height_point:
            meters = _parse_height_meters(height_point.get("height", {}))
            if meters is not None:
                snapshot["height_cm"] = round(meters * 100, 1)
                snapshot["sources"].append("google_health_height")
    except (GoogleHealthAPIError, GoogleAuthRequiredError, ValueError, TypeError) as exc:
        logger.info("Google Health body metrics unavailable: %s", exc)
        if isinstance(exc, GoogleAuthRequiredError):
            snapshot["auth_required"] = True

    if _env_override("USER_PROFILE_AGE"):
        try:
            snapshot["age_years"] = int(_env_override("USER_PROFILE_AGE"))
            snapshot["sources"].append("env_override")
        except ValueError:
            pass
    if _env_override("USER_PROFILE_HEIGHT_CM"):
        try:
            snapshot["height_cm"] = float(_env_override("USER_PROFILE_HEIGHT_CM"))
            snapshot["sources"].append("env_override")
        except ValueError:
            pass
    if _env_override("USER_PROFILE_WEIGHT_KG"):
        try:
            snapshot["weight_kg"] = float(_env_override("USER_PROFILE_WEIGHT_KG"))
            snapshot["sources"].append("env_override")
        except ValueError:
            pass
    if _env_override("USER_PROFILE_SEX"):
        snapshot["sex"] = _env_override("USER_PROFILE_SEX")
        snapshot["sources"].append("env_override")

    _cache = {
        "expires_at": now + PROFILE_CACHE_SECONDS,
        "snapshot": snapshot,
    }
    return dict(snapshot)


def format_user_profile_for_prompt(snapshot: dict[str, Any]) -> str:
    """Format profile snapshot for injection into LLM system prompts."""
    lines = [
        "USER PROFILE (use for personalized calorie burn, portion context, and coaching — do not invent missing fields):"
    ]
    if snapshot.get("age_years") is not None:
        lines.append(f"- Age: {snapshot['age_years']} years")
    if snapshot.get("height_cm") is not None:
        lines.append(f"- Height: {snapshot['height_cm']} cm")
    if snapshot.get("weight_kg") is not None:
        lines.append(f"- Latest weight: {snapshot['weight_kg']} kg")
    if snapshot.get("sex"):
        lines.append(f"- Sex: {snapshot['sex']}")
    if snapshot.get("walking_stride_mm") is not None:
        lines.append(f"- Walking stride: {snapshot['walking_stride_mm']} mm")
    if snapshot.get("running_stride_mm") is not None:
        lines.append(f"- Running stride: {snapshot['running_stride_mm']} mm")

    populated = len(lines) > 1
    if not populated:
        return (
            "USER PROFILE: No age/height/weight on file yet. "
            "Give general guidance and note that estimates vary by body size."
        )
    return "\n".join(lines)
