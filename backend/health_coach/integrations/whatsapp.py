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

    body = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
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
