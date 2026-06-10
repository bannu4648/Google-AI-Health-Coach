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
    try:
        entry = body.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None, None
        message = messages[0]
        sender = message.get("from")
        if message.get("type") != "text":
            return sender, None
        text = message.get("text", {}).get("body")
        return sender, text
    except (IndexError, KeyError, TypeError):
        return None, None
