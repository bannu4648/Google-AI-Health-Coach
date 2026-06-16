"""Async WhatsApp webhook processing."""

from __future__ import annotations

import logging
from typing import Any

from ..agent.graph import run_coach
from ..core.database import (
    get_whatsapp_reply,
    record_message,
    record_whatsapp_reply,
)
from ..integrations.google_auth import GoogleAuthRequiredError, whatsapp_reauth_message
from ..integrations.whatsapp import send_interactive_confirm_buttons, send_text_message

logger = logging.getLogger(__name__)


def _send_status(send_result: dict[str, Any]) -> str:
    if send_result.get("skipped"):
        return "skipped"
    if send_result.get("error"):
        return "error"
    return "sent"


def process_incoming_message(
    *,
    sender: str,
    message_type: str,
    message_id: str | None,
    body: dict[str, Any],
    user_text: str = "",
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    image_caption: str = "",
    document_bytes: bytes | None = None,
    document_mime_type: str | None = None,
    document_filename: str = "",
    audio_bytes: bytes | None = None,
    audio_mime_type: str | None = None,
) -> None:
    """Run coach graph and send reply; safe to call from a background task."""
    if message_id:
        existing = get_whatsapp_reply(message_id)
        if existing and existing.get("send_status") == "sent":
            logger.info("Skipping duplicate processing for message_id=%s (already sent)", message_id)
            return

    try:
        result = run_coach(
            user_text=user_text,
            sender_phone=sender,
            message_type=message_type,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            image_caption=image_caption,
            document_bytes=document_bytes,
            document_mime_type=document_mime_type,
            document_filename=document_filename,
            audio_bytes=audio_bytes,
            audio_mime_type=audio_mime_type,
        )
        final_reply = result.get("final_reply") or result.get("conversational_reply") or (
            "I heard you, but I could not produce a coach reply just now."
        )
        pending_confirm = result.get("pending_confirm", False)
        use_buttons = result.get("use_interactive_buttons", False)
        if pending_confirm or use_buttons:
            send_result = send_interactive_confirm_buttons(sender, body_text=final_reply)
        else:
            send_result = send_text_message(sender, final_reply)
        status = _send_status(send_result)
        if message_id:
            record_whatsapp_reply(
                message_id,
                reply_text=final_reply,
                send_status=status,
                phone=sender,
            )
        record_message(
            "outbound",
            phone=sender,
            text=final_reply,
            status=status,
            payload={"intent": result.get("intent"), "send_result": send_result},
            message_id=f"reply-{message_id}" if message_id else None,
        )
    except GoogleAuthRequiredError as exc:
        logger.warning("Google Health auth required: %s", exc)
        fallback = whatsapp_reauth_message(phone=sender)
        send_result = send_text_message(sender, fallback)
        if message_id:
            record_whatsapp_reply(
                message_id,
                reply_text=fallback,
                send_status=_send_status(send_result),
                phone=sender,
            )
    except Exception as exc:
        logger.exception("Background coach processing failed: %s", exc)
        fallback = "Something went wrong on my end. Please try again in a moment."
        send_result = send_text_message(sender, fallback)
        if message_id:
            record_whatsapp_reply(
                message_id,
                reply_text=fallback,
                send_status=_send_status(send_result),
                phone=sender,
            )
