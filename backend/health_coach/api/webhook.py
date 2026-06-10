"""WhatsApp Cloud API webhook routes."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from ..agent.graph import run_coach
from ..core.database import record_message
from ..integrations.whatsapp import extract_incoming_text, send_text_message

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
    sender, user_text = extract_incoming_text(body)

    if not sender:
        logger.info("Ignoring webhook without sender/message payload")
        return {"status": "ignored"}

    if not user_text:
        record_message("inbound", phone=sender, text=None, status="non_text", payload=body)
        send_text_message(
            sender,
            "I can currently understand text messages only. Send me a meal, metric, or question.",
        )
        return {"status": "ignored_non_text"}

    logger.info("Incoming message from %s: %s", sender, user_text)
    record_message("inbound", phone=sender, text=user_text, status="received", payload=body)

    result = run_coach(user_text=user_text, sender_phone=sender)
    final_reply = result.get("final_reply") or result.get("conversational_reply") or (
        "I heard you, but I could not produce a coach reply just now."
    )
    send_result = send_text_message(sender, final_reply)
    return {
        "status": "processed",
        "intent": result.get("intent"),
        "sent": send_result,
    }
