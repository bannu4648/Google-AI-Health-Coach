"""Local weight log mirror + weekly nudge helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..core.database import fetch_latest_weight_log, record_weight_log
from ..core.timezone import get_user_tz, now_local, parse_to_utc
from ..integrations.google_health import GoogleHealthAPIError, GoogleHealthClient

logger = logging.getLogger(__name__)

WEIGHT_NUDGE_INTERVAL_DAYS = 7


def invalidate_profile_cache() -> None:
    from .user_profile import invalidate_user_profile_cache

    invalidate_user_profile_cache()


def record_weight_after_sync(
    *,
    weight_kg: float,
    logged_at_hkt: str | None = None,
    source: str = "coach",
    google_health_resource: str | None = None,
    notes: str | None = None,
) -> str:
    logged = logged_at_hkt or now_local().strftime("%Y-%m-%dT%H:%M:%S")
    row_id = record_weight_log(
        weight_kg=weight_kg,
        logged_at_hkt=logged,
        source=source,
        google_health_resource=google_health_resource,
        notes=notes,
    )
    invalidate_profile_cache()
    return row_id


def _latest_google_health_weight(client: GoogleHealthClient) -> tuple[float | None, datetime | None]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=730)
    try:
        result = client.reconcile_data_points(
            "weight",
            start_time=start,
            end_time=end,
            page_size=20,
        )
    except (GoogleHealthAPIError, ValueError) as exc:
        logger.info("Could not fetch weight history: %s", exc)
        return None, None

    points = result.get("dataPoints", [])
    if not points:
        return None, None

    def _sort_key(point: dict[str, Any]) -> str:
        weight = point.get("weight", {})
        sample = weight.get("sampleTime", {})
        return str(sample.get("physicalTime") or "")

    points.sort(key=_sort_key, reverse=True)
    latest = points[0]
    grams = latest.get("weight", {}).get("weightGrams")
    if grams is None:
        return None, None
    sample_time = latest.get("weight", {}).get("sampleTime", {}).get("physicalTime")
    logged_dt: datetime | None = None
    if sample_time:
        try:
            logged_dt = parse_to_utc(sample_time).astimezone(get_user_tz())
        except (TypeError, ValueError):
            logged_dt = None
    return round(float(grams) / 1000.0, 2), logged_dt


def days_since_last_weight_log(*, client: GoogleHealthClient | None = None) -> int | None:
    """Days since the most recent weight entry (local DB or Google Health)."""
    local = fetch_latest_weight_log()
    local_dt: datetime | None = None
    if local and local.get("logged_at_hkt"):
        try:
            local_dt = parse_to_utc(str(local["logged_at_hkt"]).strip().rstrip("Z")).astimezone(
                get_user_tz()
            )
        except (TypeError, ValueError):
            local_dt = None

    remote_kg, remote_dt = _latest_google_health_weight(client or GoogleHealthClient())
    _ = remote_kg

    candidates = [dt for dt in (local_dt, remote_dt) if dt is not None]
    if not candidates:
        return None
    latest = max(candidates)
    delta = now_local() - latest
    return max(0, delta.days)


def should_send_weekly_weight_nudge(*, client: GoogleHealthClient | None = None) -> bool:
    days = days_since_last_weight_log(client=client)
    if days is None:
        return True
    return days >= WEIGHT_NUDGE_INTERVAL_DAYS


def build_weight_nudge_message(*, client: GoogleHealthClient | None = None) -> str:
    from .user_profile import fetch_user_profile_snapshot

    snapshot = fetch_user_profile_snapshot(client=client)
    last_kg = snapshot.get("weight_kg")
    days = days_since_last_weight_log(client=client)
    if last_kg is not None and days is not None:
        return (
            f"Weekly weigh-in: your last recorded weight was {last_kg} kg ({days} day(s) ago). "
            "Reply with your weight in kg (e.g. '76 kg') and I'll log it to Google Health."
        )
    return (
        "Weekly weigh-in: I don't have a recent weight on file. "
        "Reply with your weight in kg (e.g. '76 kg') and I'll log it to Google Health."
    )
