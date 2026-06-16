"""Tests for mobile Google OAuth link generation."""

from __future__ import annotations

from unittest.mock import patch

from backend.health_coach.integrations.google_auth import (
    create_mobile_auth_link,
    whatsapp_reauth_message,
)


def test_whatsapp_reauth_message_without_public_url():
    with patch("backend.health_coach.integrations.google_auth.public_base_url", return_value=None):
        msg = whatsapp_reauth_message(phone="85253016865")
    assert "python3 -m backend.health_coach.integrations.google_auth" in msg


def test_whatsapp_reauth_message_with_public_url():
    with patch("backend.health_coach.integrations.google_auth.public_base_url", return_value="https://example.ngrok.app"):
        msg = whatsapp_reauth_message(phone="85253016865")
    assert "https://example.ngrok.app/auth/google/start?state=" in msg
    assert "Tap this link" in msg


def test_create_mobile_auth_link_returns_none_without_base_url():
    with patch("backend.health_coach.integrations.google_auth.public_base_url", return_value=None):
        assert create_mobile_auth_link(phone="852") is None
