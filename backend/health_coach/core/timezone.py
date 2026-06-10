"""
User timezone helpers — defaults to Hong Kong (HKT, UTC+8).
"""

from __future__ import annotations

import copy
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

USER_TIMEZONE = ZoneInfo(os.getenv("USER_TIMEZONE", "Asia/Hong_Kong"))
USER_TIMEZONE_LABEL = os.getenv("USER_TIMEZONE_LABEL", "HKT")


def get_user_tz() -> ZoneInfo:
    return USER_TIMEZONE


def now_local() -> datetime:
    return datetime.now(USER_TIMEZONE)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def user_utc_offset_duration() -> str:
    """Google Health API duration string for the user's current UTC offset."""
    offset = now_local().utcoffset()
    if offset is None:
        return "28800s"
    return f"{int(offset.total_seconds())}s"


def format_local(dt: datetime, *, with_label: bool = True) -> str:
    local = dt.astimezone(USER_TIMEZONE)
    text = local.strftime("%Y-%m-%d %H:%M")
    return f"{text} {USER_TIMEZONE_LABEL}" if with_label else text


def format_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_to_utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=USER_TIMEZONE).astimezone(timezone.utc)
        return value.astimezone(timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=USER_TIMEZONE).astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def utc_iso_to_local_display(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return format_local(parse_to_utc(value))
    except ValueError:
        return value


def local_date_str(dt: datetime | None = None) -> str:
    return (dt or now_local()).strftime("%Y-%m-%d")


def local_datetime_str(dt: datetime | None = None) -> str:
    local = (dt or now_local()).astimezone(USER_TIMEZONE)
    if local.hour == 0 and local.minute == 0 and local.second == 0:
        return local.strftime("%Y-%m-%d")
    return local.strftime("%Y-%m-%dT%H:%M:%S")


def to_civil_filter_literal(value: datetime | str) -> str:
    """Format a UTC instant as a civil filter literal in the user's local timezone."""
    return local_datetime_str(parse_to_utc(value).astimezone(USER_TIMEZONE))


def default_query_range_utc(*, days: int = 7) -> tuple[str, str]:
    """Return UTC ISO bounds for the last N local-calendar days."""
    end_local = now_local()
    start_local = end_local - timedelta(days=days)
    return format_utc_iso(start_local), format_utc_iso(end_local)


def llm_time_context() -> str:
    """Current local time block injected into every LLM prompt."""
    local = now_local()
    utc = local.astimezone(timezone.utc)
    return (
        f"User timezone: {USER_TIMEZONE_LABEL} ({USER_TIMEZONE.key}, UTC+8)\n"
        f"Current local date/time: {format_local(local)}\n"
        f"Current UTC time: {format_utc_iso(utc)}\n"
        f"Interpret all user phrases like 'today', 'yesterday', 'morning', "
        f"'7am', and 'dinner' relative to {USER_TIMEZONE_LABEL}.\n"
        f"For meal, hydration, and weight logs, write local wall-clock timestamps "
        f"as logged_at_hkt with no timezone suffix; Python converts them to UTC.\n"
        f"For query ranges, write start_time/end_time as ISO8601 UTC with a Z suffix.\n"
        f"When writing replies to the user, always display times in {USER_TIMEZONE_LABEL}."
    )


_TIME_FIELD_RE = re.compile(
    r"^(startTime|endTime|physicalTime|createTime|updateTime|logged_at)$"
)


def enrich_health_api_result(data: Any) -> Any:
    """
    Add parallel *Local display fields beside UTC timestamps for LLM summarization.
    """
    if isinstance(data, dict):
        enriched: dict[str, Any] = {}
        for key, value in data.items():
            enriched[key] = enrich_health_api_result(value)
            if _TIME_FIELD_RE.match(key) and isinstance(value, str):
                local_key = f"{key}{USER_TIMEZONE_LABEL}"
                enriched[local_key] = utc_iso_to_local_display(value)
        if "date" in data and isinstance(data["date"], dict):
            date = data["date"]
            if all(k in date for k in ("year", "month", "day")):
                enriched[f"date{USER_TIMEZONE_LABEL}"] = (
                    f"{date['year']:04d}-{date['month']:02d}-{date['day']:02d}"
                )
        return enriched
    if isinstance(data, list):
        return [enrich_health_api_result(item) for item in data]
    return data


def enrich_health_api_result_for_llm(data: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy API payload and annotate timestamps for the summarizer."""
    return enrich_health_api_result(copy.deepcopy(data))
