"""
Meta WhatsApp Cloud API client.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

from ..core.database import record_message

load_dotenv()

logger = logging.getLogger(__name__)

WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v25.0")
WHATSAPP_COACH_TEMPLATE = os.getenv("WHATSAPP_COACH_TEMPLATE", "daily_coach_summary")
WHATSAPP_SESSION_KEEPER_TEMPLATE = os.getenv("WHATSAPP_SESSION_KEEPER_TEMPLATE", "hello_world")


def _messages_url() -> str:
    return (
        f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )


def send_text_message(recipient_phone: str, text: str) -> dict[str, Any]:
    """Send a free-form text reply."""
    if not WHATSAPP_ACCESS_TOKEN:
        logger.warning("WHATSAPP_ACCESS_TOKEN missing; skipping outbound message.")
        record_message(
            "outbound",
            phone=recipient_phone,
            text=text,
            status="skipped_missing_token",
        )
        return {"skipped": True, "reason": "missing_access_token"}
    if not WHATSAPP_PHONE_NUMBER_ID:
        logger.warning("WHATSAPP_PHONE_NUMBER_ID missing; skipping outbound message.")
        record_message(
            "outbound",
            phone=recipient_phone,
            text=text,
            status="skipped_missing_phone_number_id",
            payload={},
        )
        return {"skipped": True, "reason": "missing_phone_number_id"}

    body = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "text",
        "text": {"body": text[:4096]},
    }
    return _post_message(body)


def send_template_message(
    recipient_phone: str,
    *,
    template_name: str = "hello_world",
    language_code: str = "en_US",
    body_parameters: list[str] | None = None,
) -> dict[str, Any]:
    """Send a pre-approved WhatsApp template (e.g. hello_world)."""
    if not WHATSAPP_ACCESS_TOKEN:
        logger.warning("WHATSAPP_ACCESS_TOKEN missing; skipping outbound message.")
        record_message(
            "outbound",
            phone=recipient_phone,
            text=f"template:{template_name}",
            status="skipped_missing_token",
        )
        return {"skipped": True, "reason": "missing_access_token"}
    if not WHATSAPP_PHONE_NUMBER_ID:
        logger.warning("WHATSAPP_PHONE_NUMBER_ID missing; skipping template message.")
        record_message(
            "outbound",
            phone=recipient_phone,
            text=f"template:{template_name}",
            status="skipped_missing_phone_number_id",
        )
        return {"skipped": True, "reason": "missing_phone_number_id"}

    template_block: dict[str, Any] = {
        "name": template_name,
        "language": {"code": language_code},
    }
    if body_parameters:
        template_block["components"] = [
            {
                "type": "body",
                "parameters": [{"type": "text", "text": text[:1024]} for text in body_parameters],
            }
        ]

    body = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "template",
        "template": template_block,
    }
    return _post_message(body)


def send_coach_summary_template(recipient_phone: str, summary_text: str) -> dict[str, Any]:
    """Send coach summary via approved template (fallback when 24h window closed)."""
    snippet = summary_text[:900].strip()
    if not snippet:
        snippet = "Your health coach summary is ready. Open WhatsApp to view details."
    return send_template_message(
        recipient_phone,
        template_name=WHATSAPP_COACH_TEMPLATE,
        body_parameters=[snippet],
    )


def send_interactive_confirm_buttons(
    recipient_phone: str,
    *,
    body_text: str,
    confirm_id: str = "confirm_log",
    skip_id: str = "skip_log",
) -> dict[str, Any]:
    """Send Confirm / Skip quick-reply buttons for low-confidence nutrition logs."""
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        return send_text_message(
            recipient_phone,
            f"{body_text}\n\nReply 'yes' to log or 'skip' to cancel.",
        )
    body = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text[:1024]},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": confirm_id, "title": "Log it"}},
                    {"type": "reply", "reply": {"id": skip_id, "title": "Skip"}},
                ]
            },
        },
    }
    return _post_message(body)


def _post_message(body: dict[str, Any]) -> dict[str, Any]:
    recipient = body.get("to")
    message_text = body.get("text", {}).get("body") or f"template:{body.get('template', {}).get('name', '')}"
    try:
        response = requests.post(
            _messages_url(),
            headers={
                "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()
        record_message(
            "outbound",
            phone=recipient,
            text=message_text,
            status="sent",
            payload={"request": body, "response": result},
        )
        return result
    except requests.RequestException as exc:
        detail = ""
        meta_message = ""
        if exc.response is not None:
            detail = exc.response.text
            try:
                meta_message = exc.response.json().get("error", {}).get("message", "")
            except ValueError:
                pass
        if exc.response is not None and exc.response.status_code == 401:
            logger.error(
                "WhatsApp token rejected (401). %s "
                "Generate a new token in Meta Developer Console → WhatsApp → API Setup.",
                meta_message or detail,
            )
        else:
            logger.exception("Failed to send WhatsApp message: %s", exc)
        result = {
            "error": True,
            "message": meta_message or str(exc),
            "detail": detail,
            "status_code": getattr(exc.response, "status_code", None),
        }
        record_message(
            "outbound",
            phone=recipient,
            text=message_text,
            status="error",
            payload={"request": body, "response": result},
        )
        return result


def extract_incoming_text(body: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (sender_phone, message_text) from a Meta webhook payload."""
    incoming = extract_incoming_message(body)
    if not incoming:
        return None, None
    if incoming["message_type"] != "text":
        return incoming["sender"], None
    return incoming["sender"], incoming.get("text")


def extract_incoming_message(body: dict[str, Any]) -> dict[str, Any] | None:
    """Return normalized inbound message metadata from a Meta webhook payload."""
    try:
        entry = body.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None
        message = messages[0]
        sender = message.get("from")
        message_type = message.get("type", "unknown")
        result: dict[str, Any] = {
            "sender": sender,
            "message_type": message_type,
            "message_id": message.get("id"),
            "text": None,
            "media_id": None,
            "mime_type": None,
            "caption": "",
        }
        if message_type == "text":
            result["text"] = message.get("text", {}).get("body")
            return result
        if message_type == "image":
            image = message.get("image", {})
            result["media_id"] = image.get("id")
            result["mime_type"] = image.get("mime_type", "image/jpeg")
            result["caption"] = image.get("caption") or ""
            return result
        if message_type == "audio":
            audio = message.get("audio", {})
            result["media_id"] = audio.get("id")
            result["mime_type"] = audio.get("mime_type", "audio/ogg")
            return result
        if message_type == "document":
            document = message.get("document", {})
            result["media_id"] = document.get("id")
            result["mime_type"] = document.get("mime_type", "application/pdf")
            result["filename"] = document.get("filename") or "document"
            result["caption"] = document.get("caption") or ""
            return result
        return result
    except (IndexError, KeyError, TypeError):
        return None


def download_whatsapp_media(media_id: str) -> tuple[bytes, str]:
    """Download WhatsApp media bytes and mime type via the Cloud API."""
    if not WHATSAPP_ACCESS_TOKEN:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN missing; cannot download media.")
    meta_url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{media_id}"
    meta_response = requests.get(
        meta_url,
        headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"},
        timeout=20,
    )
    meta_response.raise_for_status()
    meta = meta_response.json()
    download_url = meta.get("url")
    mime_type = meta.get("mime_type", "image/jpeg")
    if not download_url:
        raise RuntimeError(f"WhatsApp media metadata missing url for {media_id}")

    file_response = requests.get(
        download_url,
        headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"},
        timeout=30,
    )
    file_response.raise_for_status()
    return file_response.content, mime_type
