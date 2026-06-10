"""Local scheduled coach messages."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from ..core.database import record_job_run, utc_now_iso
from ..core.timezone import get_user_tz
from ..integrations.whatsapp import send_template_message, send_text_message
from .coaching import create_daily_summary

load_dotenv()

logger = logging.getLogger(__name__)

ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"
MORNING_SUMMARY_TIME = os.getenv("MORNING_SUMMARY_TIME", "08:00")
EVENING_SUMMARY_TIME = os.getenv("EVENING_SUMMARY_TIME", "21:30")
SUMMARY_RECIPIENT_PHONE = os.getenv("SUMMARY_RECIPIENT_PHONE", "")

_scheduler: BackgroundScheduler | None = None


def _parse_hhmm(value: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected HH:MM time, got {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Expected HH:MM time in 24-hour range, got {value!r}")
    return hour, minute


def _send_with_template_fallback(phone: str, message: str) -> dict:
    result = send_text_message(phone, message)
    if result.get("error"):
        template = send_template_message(phone, template_name="hello_world")
        return {"text_result": result, "template_result": template}
    return {"text_result": result}


def run_summary_job(summary_type: str) -> dict:
    started = utc_now_iso()
    try:
        if not SUMMARY_RECIPIENT_PHONE:
            raise RuntimeError("SUMMARY_RECIPIENT_PHONE is not set")
        summary = create_daily_summary(summary_type)
        send_result = _send_with_template_fallback(
            SUMMARY_RECIPIENT_PHONE,
            summary["message"],
        )
        result = {**summary, "send_result": send_result}
        record_job_run(
            job_name=f"{summary_type}_summary",
            status="success",
            started_at=started,
            finished_at=utc_now_iso(),
            result=result,
        )
        return result
    except Exception as exc:
        logger.exception("Scheduled %s summary failed: %s", summary_type, exc)
        record_job_run(
            job_name=f"{summary_type}_summary",
            status="error",
            started_at=started,
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        return {"error": True, "message": str(exc)}


def start_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if not ENABLE_SCHEDULER:
        logger.info("Scheduler disabled. Set ENABLE_SCHEDULER=true to enable.")
        return None
    if _scheduler and _scheduler.running:
        return _scheduler

    scheduler = BackgroundScheduler(timezone=get_user_tz())
    try:
        morning_hour, morning_minute = _parse_hhmm(MORNING_SUMMARY_TIME)
        evening_hour, evening_minute = _parse_hhmm(EVENING_SUMMARY_TIME)
    except ValueError as exc:
        logger.error("Scheduler disabled because summary time config is invalid: %s", exc)
        return None

    scheduler.add_job(
        run_summary_job,
        CronTrigger(hour=morning_hour, minute=morning_minute, timezone=get_user_tz()),
        id="morning_summary",
        args=["morning"],
        replace_existing=True,
    )
    scheduler.add_job(
        run_summary_job,
        CronTrigger(hour=evening_hour, minute=evening_minute, timezone=get_user_tz()),
        id="evening_summary",
        args=["evening"],
        replace_existing=True,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Scheduler started at %s with morning=%s evening=%s",
        datetime.now(get_user_tz()).isoformat(),
        MORNING_SUMMARY_TIME,
        EVENING_SUMMARY_TIME,
    )
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
