"""WhatsApp Cloud API webhook routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from ..core.database import claim_whatsapp_message, hash_webhook_body, record_message
from ..integrations.whatsapp import (
    download_whatsapp_media,
    extract_incoming_message,
    send_text_message,
)
from .webhook_processor import process_incoming_message

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhook"])
VERIFY_TOKEN = __import__("os").getenv("WHATSAPP_VERIFY_TOKEN", "health-coach-verify-token")


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


def _enqueue_message(
    background_tasks: BackgroundTasks,
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
) -> dict[str, Any]:
    background_tasks.add_task(
        process_incoming_message,
        sender=sender,
        message_type=message_type,
        message_id=message_id,
        body=body,
        user_text=user_text,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        image_caption=image_caption,
        document_bytes=document_bytes,
        document_mime_type=document_mime_type,
        document_filename=document_filename,
        audio_bytes=audio_bytes,
        audio_mime_type=audio_mime_type,
    )
    return {"status": "accepted", "message_id": message_id}


@router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Receive WhatsApp messages and process coach replies asynchronously."""
    body = await request.json()
    incoming = extract_incoming_message(body)
    if not incoming or not incoming.get("sender"):
        logger.info("Ignoring webhook without sender/message payload")
        return {"status": "ignored"}

    sender = incoming["sender"]
    message_type = incoming.get("message_type", "unknown")
    message_id = incoming.get("message_id")
    body_hash = hash_webhook_body(body) if not message_id else None

    if not claim_whatsapp_message(message_id, body_hash=body_hash):
        logger.info(
            "Ignoring duplicate webhook for message_id=%s from %s",
            message_id or body_hash,
            sender,
        )
        return {"status": "duplicate"}

    if message_type == "text":
        user_text = incoming.get("text") or ""
        logger.info("Incoming text from %s: %s", sender, user_text)
        record_message(
            "inbound",
            phone=sender,
            text=user_text,
            status="received",
            payload=body,
            message_id=message_id,
        )
        return _enqueue_message(
            background_tasks,
            sender=sender,
            message_type="text",
            message_id=message_id,
            body=body,
            user_text=user_text,
        )

    if message_type == "image":
        media_id = incoming.get("media_id")
        caption = incoming.get("caption") or ""
        logger.info("Incoming image from %s (caption=%r, media_id=%s)", sender, caption, media_id)
        record_message(
            "inbound",
            phone=sender,
            text=caption or "[image]",
            status="received_image",
            payload=body,
            message_id=message_id,
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
        return _enqueue_message(
            background_tasks,
            sender=sender,
            message_type="image",
            message_id=message_id,
            body=body,
            user_text=caption,
            image_bytes=image_bytes,
            image_mime_type=mime_type,
            image_caption=caption,
        )

    if message_type == "audio":
        media_id = incoming.get("media_id")
        logger.info("Incoming audio from %s (media_id=%s)", sender, media_id)
        record_message(
            "inbound",
            phone=sender,
            text="[audio]",
            status="received_audio",
            payload=body,
            message_id=message_id,
        )
        try:
            audio_bytes, mime_type = download_whatsapp_media(media_id)
        except Exception as exc:
            logger.exception("Failed to download WhatsApp audio: %s", exc)
            send_text_message(sender, "I couldn't download that voice note. Please try again.")
            return {"status": "audio_download_failed"}
        return _enqueue_message(
            background_tasks,
            sender=sender,
            message_type="audio",
            message_id=message_id,
            body=body,
            audio_bytes=audio_bytes,
            audio_mime_type=mime_type,
        )

    if message_type == "document":
        media_id = incoming.get("media_id")
        filename = incoming.get("filename") or "document"
        caption = incoming.get("caption") or ""
        logger.info("Incoming document from %s (%s)", sender, filename)
        record_message(
            "inbound",
            phone=sender,
            text=caption or f"[document:{filename}]",
            status="received_document",
            payload=body,
            message_id=message_id,
        )
        try:
            document_bytes, mime_type = download_whatsapp_media(media_id)
        except Exception as exc:
            logger.exception("Failed to download WhatsApp document: %s", exc)
            send_text_message(sender, "I couldn't download that file. Please try again.")
            return {"status": "document_download_failed"}
        return _enqueue_message(
            background_tasks,
            sender=sender,
            message_type="document",
            message_id=message_id,
            body=body,
            user_text=caption,
            document_bytes=document_bytes,
            document_mime_type=mime_type,
            document_filename=filename,
        )

    record_message(
        "inbound",
        phone=sender,
        text=None,
        status="unsupported_type",
        payload=body,
        message_id=message_id,
    )
    send_text_message(
        sender,
        "I can handle text, voice notes, food photos, and PDF/documents. "
        "Send a message describing what you ate or ask a health question.",
    )
    return {"status": "ignored_non_text"}
