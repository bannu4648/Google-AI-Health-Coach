"""
Local Google OAuth authorization for the Google Health API v4.

Re-authorize when refresh fails (invalid_grant / revoked token):
    python3 -m backend.health_coach.integrations.google_auth

Normal access-token expiry is refreshed automatically — no browser needed.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow

from ..core.database import (
    create_oauth_pending_state,
    get_oauth_pending_state,
    mark_oauth_pending_state_used,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"

HEALTH_SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.writeonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.writeonly",
    "https://www.googleapis.com/auth/googlehealth.nutrition.readonly",
    "https://www.googleapis.com/auth/googlehealth.nutrition.writeonly",
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
    "https://www.googleapis.com/auth/googlehealth.sleep.writeonly",
    "https://www.googleapis.com/auth/googlehealth.profile.readonly",
]

REAUTH_COMMAND = "python3 -m backend.health_coach.integrations.google_auth"
OAUTH_STATE_TTL_MINUTES = 15


class GoogleAuthRequiredError(Exception):
    """Raised when Google OAuth cannot be refreshed and browser re-consent is required."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message
            or (
                "Google Health authorization expired or was revoked. "
                f"On the machine running the coach, run:\n{REAUTH_COMMAND}"
            )
        )


def _save_token(creds: Credentials, token_path: Path) -> None:
    token_path.write_text(creds.to_json(), encoding="utf-8")
    logger.info("Saved credentials to %s", token_path)


def _try_refresh(creds: Credentials, token_path: Path) -> Credentials | None:
    """Refresh access token. Returns None if refresh_token is revoked (needs browser)."""
    if not creds.refresh_token:
        return None
    try:
        creds.refresh(Request())
        _save_token(creds, token_path)
        return creds
    except RefreshError as exc:
        logger.warning("OAuth refresh failed (%s) — browser re-consent required.", exc)
        return None


def public_base_url() -> str | None:
    """Public HTTPS base URL (ngrok or production) for mobile OAuth callbacks."""
    raw = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    return raw or None


def oauth_callback_uri() -> str | None:
    base = public_base_url()
    if not base:
        return None
    return f"{base}/auth/google/callback"


def _load_client_config(credentials_path: Path = CREDENTIALS_FILE) -> dict:
    """Prefer explicit `web` client for mobile OAuth; fall back to installed desktop client."""
    if not credentials_path.exists():
        raise FileNotFoundError(f"Missing OAuth client secrets at {credentials_path}")
    data = json.loads(credentials_path.read_text(encoding="utf-8"))
    if "web" in data:
        return {"web": data["web"]}
    if "installed" in data:
        installed = data["installed"]
        return {
            "web": {
                "client_id": installed["client_id"],
                "client_secret": installed["client_secret"],
                "auth_uri": installed.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": installed.get("token_uri", "https://oauth2.googleapis.com/token"),
            }
        }
    raise ValueError("credentials.json must contain 'web' or 'installed' client config.")


def _installed_client_config(credentials_path: Path = CREDENTIALS_FILE) -> dict | None:
    if not credentials_path.exists():
        return None
    data = json.loads(credentials_path.read_text(encoding="utf-8"))
    if "installed" in data:
        return data
    return None


def _build_web_flow(*, redirect_uri: str, state: str | None = None) -> Flow:
    return Flow.from_client_config(
        _load_client_config(),
        scopes=HEALTH_SCOPES,
        redirect_uri=redirect_uri,
        state=state,
    )


def create_mobile_auth_link(*, phone: str | None = None) -> str | None:
    """
    Short HTTPS link the user opens on their phone to re-authorize Google Health.

    Requires PUBLIC_BASE_URL and callback URI registered in Google Cloud Console.
    """
    base = public_base_url()
    if not base:
        return None
    state = secrets.token_urlsafe(24)
    expires = (datetime.now(timezone.utc) + timedelta(minutes=OAUTH_STATE_TTL_MINUTES)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    create_oauth_pending_state(phone=phone, expires_at=expires, state=state)
    return f"{base}/auth/google/start?state={state}"


def build_google_authorization_url(*, state: str) -> str:
    redirect_uri = oauth_callback_uri()
    if not redirect_uri:
        raise GoogleAuthRequiredError("PUBLIC_BASE_URL is not configured for mobile OAuth.")
    pending = get_oauth_pending_state(state)
    if not pending or pending.get("used_at"):
        raise GoogleAuthRequiredError("This sign-in link expired or was already used.")
    if pending["expires_at"] < utc_now_iso():
        raise GoogleAuthRequiredError("This sign-in link expired. Message the coach for a new link.")
    flow = _build_web_flow(redirect_uri=redirect_uri, state=state)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return auth_url


def complete_mobile_auth(*, state: str, code: str) -> Credentials:
    redirect_uri = oauth_callback_uri()
    if not redirect_uri:
        raise GoogleAuthRequiredError("PUBLIC_BASE_URL is not configured.")
    pending = get_oauth_pending_state(state)
    if not pending or pending.get("used_at"):
        raise GoogleAuthRequiredError("Invalid or used OAuth state.")
    if pending["expires_at"] < utc_now_iso():
        raise GoogleAuthRequiredError("OAuth link expired.")
    flow = _build_web_flow(redirect_uri=redirect_uri, state=state)
    flow.fetch_token(code=code)
    creds = flow.credentials
    if not creds or not creds.valid:
        raise GoogleAuthRequiredError("Google did not return valid credentials.")
    _save_token(creds, TOKEN_FILE)
    mark_oauth_pending_state_used(state)
    logger.info("Mobile OAuth complete; token.json updated.")
    return creds


def whatsapp_reauth_message(*, phone: str | None = None) -> str:
    """WhatsApp text with a mobile sign-in link when PUBLIC_BASE_URL is configured."""
    link = create_mobile_auth_link(phone=phone)
    if link:
        return (
            "My Google Health connection needs to be renewed. "
            "Tap this link on your phone, sign in with Google, then message me again:\n\n"
            f"{link}\n\n"
            "(Link expires in 15 minutes.)"
        )
    return (
        "My Google Health connection needs to be renewed on the coach computer. "
        f"Run this command there, sign in in the browser, then message me again:\n\n"
        f"{REAUTH_COMMAND}"
    )


def load_credentials(
    *,
    credentials_path: Path = CREDENTIALS_FILE,
    token_path: Path = TOKEN_FILE,
    scopes: list[str] | None = None,
) -> Credentials:
    """
    Load OAuth credentials from token.json, refreshing automatically when expired.

    Raises GoogleAuthRequiredError when token.json is missing or refresh_token is revoked.
    Run this module directly (`python3 -m ...google_auth`) to re-authorize in the browser.
    """
    scopes = scopes or HEALTH_SCOPES
    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if creds and creds.expired:
        refreshed = _try_refresh(creds, token_path)
        if refreshed:
            creds = refreshed
        else:
            creds = None

    if creds and creds.valid:
        return creds

    raise GoogleAuthRequiredError(
        f"No valid Google Health token at {token_path}. "
        f"Run `{REAUTH_COMMAND}` on the coach machine to sign in again."
    )


def authorize(
    *,
    credentials_path: Path = CREDENTIALS_FILE,
    token_path: Path = TOKEN_FILE,
    port: int = 8080,
) -> Credentials:
    """
    Perform the local OAuth consent flow and persist token.json.

    Uses run_local_server to open a browser window when refresh is not possible.
    """
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Missing OAuth client secrets at {credentials_path}. "
            "Download your Desktop Client credentials from Google Cloud Console."
        )

    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), HEALTH_SCOPES)
        if creds and creds.valid:
            logger.info("Existing token.json is still valid; skipping browser flow.")
            return creds
        if creds and creds.expired:
            refreshed = _try_refresh(creds, token_path)
            if refreshed:
                logger.info("Refreshed expired token without browser.")
                return refreshed
            logger.info("Refresh token revoked — opening browser for new consent.")

    installed = _installed_client_config(credentials_path)
    if installed:
        flow = InstalledAppFlow.from_client_config(installed, scopes=HEALTH_SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes=HEALTH_SCOPES)
    creds = flow.run_local_server(port=port, prompt="consent")
    _save_token(creds, token_path)
    logger.info("Authorization complete. token.json is ready for the backend.")
    return creds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    authorize()
