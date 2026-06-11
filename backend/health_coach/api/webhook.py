"""WhatsApp Cloud API webhook routes."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from ..agent.graph import run_coach
from ..core.database import record_message
from ..integrations.whatsapp import (
    download_whatsapp_media,
    extract_incoming_message,
    send_text_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhook"])
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "health-coach-verify-token")


@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
) -> PlainTextResponse:
    """Handle Meta's webhook verification challenge."""
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403, detail="Invalid verify token")


@router.post("/webhook")
async def receive_webhook(request: Request) -> dict[str, Any]:
    """Receive WhatsApp messages, run the coach agent, and send the reply."""
    body = await request.json()
    incoming = extract_incoming_message(body)
    if not incoming or not incoming.get("sender"):
        logger.info("Ignoring webhook without sender/message payload")
        return {"status": "ignored"}

    sender = incoming["sender"]
    message_type = incoming.get("message_type", "unknown")

    if message_type == "text":
        user_text = incoming.get("text") or ""
        logger.info("Incoming text from %s: %s", sender, user_text)
        record_message("inbound", phone=sender, text=user_text, status="received", payload=body)
        result = run_coach(user_text=user_text, sender_phone=sender, message_type="text")
    elif message_type == "image":
        media_id = incoming.get("media_id")
        caption = incoming.get("caption") or ""
        logger.info("Incoming image from %s (caption=%r, media_id=%s)", sender, caption, media_id)
        record_message(
            "inbound",
            phone=sender,
            text=caption or "[image]",
            status="received_image",
            payload=body,
        )
        try:
            image_bytes, mime_type = download_whatsapp_media(media_id)
        except Exception as exc:
            logger.exception("Failed to download WhatsApp image: %s", exc)
            send_text_message(
                sender,
                "I couldn't download that photo. Please try sending it again.",
            )
            return {"status": "image_download_failed"}
        result = run_coach(
            user_text=caption,
            sender_phone=sender,
            message_type="image",
            image_bytes=image_bytes,
            image_mime_type=mime_type,
            image_caption=caption,
        )
    else:
        record_message("inbound", phone=sender, text=None, status="unsupported_type", payload=body)
        send_text_message(
            sender,
            "I can handle text and food photos right now. Send a meal photo or describe what you ate.",
        )
        return {"status": "ignored_non_text"}
    final_reply = result.get("final_reply") or result.get("conversational_reply") or (
        "I heard you, but I could not produce a coach reply just now."
    )
    send_result = send_text_message(sender, final_reply)
    return {
        "status": "processed",
        "intent": result.get("intent"),
        "sent": send_result,
    }
