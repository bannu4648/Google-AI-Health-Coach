"""Local scheduled coach messages."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from ..integrations.google_auth import GoogleAuthRequiredError, notify_google_auth_required
from ..core.database import fetch_recent_messages_for_phone, record_job_run, utc_now_iso
from ..core.timezone import get_user_tz, now_local
from ..integrations.whatsapp import (
    WHATSAPP_SESSION_KEEPER_TEMPLATE,
    send_coach_summary_template,
    send_template_message,
    send_text_message,
)
from .coaching import create_daily_summary, create_weekly_recap
from .fitness_plans import get_todays_workout
from .memory import record_coach_outreach

load_dotenv()

logger = logging.getLogger(__name__)

ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"
MORNING_SUMMARY_TIME = os.getenv("MORNING_SUMMARY_TIME", "08:00")
EVENING_SUMMARY_TIME = os.getenv("EVENING_SUMMARY_TIME", "21:30")
SUMMARY_RECIPIENT_PHONE = os.getenv("SUMMARY_RECIPIENT_PHONE", "")
READINESS_NUDGE_TIME = os.getenv("READINESS_NUDGE_TIME", "")
WORKOUT_NUDGE_TIME = os.getenv("WORKOUT_NUDGE_TIME", "18:00")
WEIGHT_LOG_NUDGE_DAY = os.getenv("WEIGHT_LOG_NUDGE_DAY", "mon").lower()
WEIGHT_LOG_NUDGE_TIME = os.getenv("WEIGHT_LOG_NUDGE_TIME", "09:00")
WEEKLY_RECAP_DAY = os.getenv("WEEKLY_RECAP_DAY", "sun").lower()
WEEKLY_RECAP_TIME = os.getenv("WEEKLY_RECAP_TIME", "21:00")
SESSION_KEEPER_HOURS = int(os.getenv("SESSION_KEEPER_HOURS", "20"))

_scheduler: BackgroundScheduler | None = None

_DAY_MAP = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def _parse_hhmm(value: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected HH:MM time, got {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Expected HH:MM time in 24-hour range, got {value!r}")
    return hour, minute


def _send_with_template_fallback(phone: str, message: str, *, job: str = "") -> dict:
    result = send_text_message(phone, message)
    if result.get("error"):
        template = send_coach_summary_template(phone, message)
        if template.get("error"):
            template = send_template_message(
                phone,
                template_name=WHATSAPP_SESSION_KEEPER_TEMPLATE,
            )
        return {"text_result": result, "template_result": template}
    record_coach_outreach(phone, text=message, source=job or "scheduler")
    return {"text_result": result}


def _maybe_session_keeper(phone: str) -> dict | None:
    """Lightweight template ping if no inbound message in N hours (keeps 24h window)."""
    if SESSION_KEEPER_HOURS <= 0:
        return None
    recent = fetch_recent_messages_for_phone(phone, limit=5)
    inbound = [row for row in recent if row.get("direction") == "inbound"]
    if not inbound:
        return send_template_message(phone, template_name=WHATSAPP_SESSION_KEEPER_TEMPLATE)
    return None


def run_readiness_nudge_job() -> dict:
    started = utc_now_iso()
    try:
        if not SUMMARY_RECIPIENT_PHONE:
            raise RuntimeError("SUMMARY_RECIPIENT_PHONE is not set")
        from .coaching import get_daily_health_snapshot

        _maybe_session_keeper(SUMMARY_RECIPIENT_PHONE)
        snapshot = get_daily_health_snapshot()
        readiness = snapshot.get("readiness", {})
        message = (
            f"Mid-day check-in: readiness {readiness.get('score', '?')}/100 "
            f"({readiness.get('label', 'steady')}). "
            + " ".join(snapshot.get("recommendations", [])[:2])
        )
        send_result = _send_with_template_fallback(
            SUMMARY_RECIPIENT_PHONE,
            message,
            job="readiness_nudge",
        )
        result = {"snapshot": snapshot, "message": message, "send_result": send_result}
        record_job_run(
            job_name="readiness_nudge",
            status="success",
            started_at=started,
            finished_at=utc_now_iso(),
            result=result,
        )
        return result
    except Exception as exc:
        logger.exception("Readiness nudge failed: %s", exc)
        record_job_run(
            job_name="readiness_nudge",
            status="error",
            started_at=started,
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        return {"error": True, "message": str(exc)}


def run_workout_adherence_nudge_job() -> dict:
    """Nudge when today's planned workout exists but no exercise logged yet."""
    started = utc_now_iso()
    try:
        if not SUMMARY_RECIPIENT_PHONE:
            raise RuntimeError("SUMMARY_RECIPIENT_PHONE is not set")
        from .coaching import get_daily_health_snapshot
        from ..services.fitness_plans import format_workout_for_reply

        today_workout = get_todays_workout()
        if not today_workout:
            return {"skipped": True, "reason": "no_planned_workout"}

        snapshot = get_daily_health_snapshot()
        if snapshot.get("exercise", {}).get("count", 0) > 0:
            return {"skipped": True, "reason": "exercise_already_logged"}

        message = (
            f"Workout reminder: {today_workout.get('title', 'session')} is on your plan today.\n"
            f"{format_workout_for_reply(today_workout)}"
        )
        send_result = _send_with_template_fallback(
            SUMMARY_RECIPIENT_PHONE,
            message,
            job="workout_adherence_nudge",
        )
        result = {"message": message, "send_result": send_result}
        record_job_run(
            job_name="workout_adherence_nudge",
            status="success",
            started_at=started,
            finished_at=utc_now_iso(),
            result=result,
        )
        return result
    except Exception as exc:
        logger.exception("Workout adherence nudge failed: %s", exc)
        record_job_run(
            job_name="workout_adherence_nudge",
            status="error",
            started_at=started,
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        return {"error": True, "message": str(exc)}


def run_weekly_weight_nudge_job() -> dict:
    """Remind user to log weight if none recorded in the past week."""
    started = utc_now_iso()
    try:
        if not SUMMARY_RECIPIENT_PHONE:
            raise RuntimeError("SUMMARY_RECIPIENT_PHONE is not set")
        from .weight_tracking import build_weight_nudge_message, should_send_weekly_weight_nudge

        if not should_send_weekly_weight_nudge():
            return {"skipped": True, "reason": "weight_logged_recently"}

        message = build_weight_nudge_message()
        send_result = _send_with_template_fallback(
            SUMMARY_RECIPIENT_PHONE,
            message,
            job="weekly_weight_nudge",
        )
        result = {"message": message, "send_result": send_result}
        record_job_run(
            job_name="weekly_weight_nudge",
            status="success",
            started_at=started,
            finished_at=utc_now_iso(),
            result=result,
        )
        return result
    except Exception as exc:
        logger.exception("Weekly weight nudge failed: %s", exc)
        record_job_run(
            job_name="weekly_weight_nudge",
            status="error",
            started_at=started,
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        return {"error": True, "message": str(exc)}


def run_weekly_recap_job() -> dict:
    started = utc_now_iso()
    try:
        if not SUMMARY_RECIPIENT_PHONE:
            raise RuntimeError("SUMMARY_RECIPIENT_PHONE is not set")
        recap = create_weekly_recap()
        send_result = _send_with_template_fallback(
            SUMMARY_RECIPIENT_PHONE,
            recap["message"],
            job="weekly_recap",
        )
        result = {**recap, "send_result": send_result}
        record_job_run(
            job_name="weekly_recap",
            status="success",
            started_at=started,
            finished_at=utc_now_iso(),
            result=result,
        )
        return result
    except Exception as exc:
        logger.exception("Weekly recap failed: %s", exc)
        record_job_run(
            job_name="weekly_recap",
            status="error",
            started_at=started,
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        return {"error": True, "message": str(exc)}


def run_summary_job(summary_type: str) -> dict:
    started = utc_now_iso()
    try:
        if not SUMMARY_RECIPIENT_PHONE:
            raise RuntimeError("SUMMARY_RECIPIENT_PHONE is not set")
        _maybe_session_keeper(SUMMARY_RECIPIENT_PHONE)
        summary = create_daily_summary(summary_type)
        send_result = _send_with_template_fallback(
            SUMMARY_RECIPIENT_PHONE,
            summary["message"],
            job=f"{summary_type}_summary",
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
    except GoogleAuthRequiredError as exc:
        logger.warning("Scheduled %s summary needs Google re-auth: %s", summary_type, exc)
        notify_google_auth_required(phone=SUMMARY_RECIPIENT_PHONE)
        record_job_run(
            job_name=f"{summary_type}_summary",
            status="error",
            started_at=started,
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        return {"error": True, "message": str(exc)}
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


def get_scheduler_config() -> dict[str, str]:
    return {
        "morning_summary_time": MORNING_SUMMARY_TIME,
        "evening_summary_time": EVENING_SUMMARY_TIME,
        "readiness_nudge_time": READINESS_NUDGE_TIME or "",
        "workout_nudge_time": WORKOUT_NUDGE_TIME,
        "weight_log_nudge_day": WEIGHT_LOG_NUDGE_DAY,
        "weight_log_nudge_time": WEIGHT_LOG_NUDGE_TIME,
        "weekly_recap_day": WEEKLY_RECAP_DAY,
        "weekly_recap_time": WEEKLY_RECAP_TIME,
        "enabled": str(ENABLE_SCHEDULER).lower(),
    }


def start_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if not ENABLE_SCHEDULER:
        logger.info("Scheduler disabled. Set ENABLE_SCHEDULER=true to enable.")
        return None
    if not SUMMARY_RECIPIENT_PHONE:
        logger.error("Scheduler disabled: SUMMARY_RECIPIENT_PHONE is not set.")
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
    if READINESS_NUDGE_TIME.strip():
        try:
            nudge_hour, nudge_minute = _parse_hhmm(READINESS_NUDGE_TIME.strip())
            scheduler.add_job(
                run_readiness_nudge_job,
                CronTrigger(hour=nudge_hour, minute=nudge_minute, timezone=get_user_tz()),
                id="readiness_nudge",
                replace_existing=True,
            )
        except ValueError as exc:
            logger.warning("Invalid READINESS_NUDGE_TIME %r: %s", READINESS_NUDGE_TIME, exc)
    if WORKOUT_NUDGE_TIME.strip():
        try:
            workout_hour, workout_minute = _parse_hhmm(WORKOUT_NUDGE_TIME.strip())
            scheduler.add_job(
                run_workout_adherence_nudge_job,
                CronTrigger(hour=workout_hour, minute=workout_minute, timezone=get_user_tz()),
                id="workout_adherence_nudge",
                replace_existing=True,
            )
        except ValueError as exc:
            logger.warning("Invalid WORKOUT_NUDGE_TIME %r: %s", WORKOUT_NUDGE_TIME, exc)
    weight_dow = _DAY_MAP.get(WEIGHT_LOG_NUDGE_DAY, 0)
    if WEIGHT_LOG_NUDGE_TIME.strip():
        try:
            weight_hour, weight_minute = _parse_hhmm(WEIGHT_LOG_NUDGE_TIME.strip())
            scheduler.add_job(
                run_weekly_weight_nudge_job,
                CronTrigger(
                    day_of_week=weight_dow,
                    hour=weight_hour,
                    minute=weight_minute,
                    timezone=get_user_tz(),
                ),
                id="weekly_weight_nudge",
                replace_existing=True,
            )
        except ValueError as exc:
            logger.warning("Invalid WEIGHT_LOG_NUDGE_TIME %r: %s", WEIGHT_LOG_NUDGE_TIME, exc)
    weekly_dow = _DAY_MAP.get(WEEKLY_RECAP_DAY, 6)
    try:
        recap_hour, recap_minute = _parse_hhmm(WEEKLY_RECAP_TIME)
        scheduler.add_job(
            run_weekly_recap_job,
            CronTrigger(
                day_of_week=weekly_dow,
                hour=recap_hour,
                minute=recap_minute,
                timezone=get_user_tz(),
            ),
            id="weekly_recap",
            replace_existing=True,
        )
    except ValueError as exc:
        logger.warning("Invalid WEEKLY_RECAP_TIME %r: %s", WEEKLY_RECAP_TIME, exc)

    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Scheduler started at %s with morning=%s evening=%s workout_nudge=%s weight_nudge=%s %s",
        datetime.now(get_user_tz()).isoformat(),
        MORNING_SUMMARY_TIME,
        EVENING_SUMMARY_TIME,
        WORKOUT_NUDGE_TIME,
        WEIGHT_LOG_NUDGE_DAY,
        WEIGHT_LOG_NUDGE_TIME,
    )
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
