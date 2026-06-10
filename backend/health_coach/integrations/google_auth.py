"""
Local Google OAuth authorization for the Google Health API v4.

Run once to generate token.json:
    python3 -m backend.health_coach.integrations.google_auth
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

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


def load_credentials(
    *,
    credentials_path: Path = CREDENTIALS_FILE,
    token_path: Path = TOKEN_FILE,
    scopes: list[str] | None = None,
) -> Credentials:
    """
    Load OAuth credentials from token.json, refreshing automatically when expired.

    If token.json is missing or invalid, raises FileNotFoundError with guidance
    to run this module directly for first-time authorization.
    """
    scopes = scopes or HEALTH_SCOPES
    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if creds and creds.expired and creds.refresh_token:
        logger.info("Access token expired; refreshing via refresh_token.")
        creds.refresh(Request())
        _save_token(creds, token_path)

    if creds and creds.valid:
        return creds

    raise FileNotFoundError(
        f"No valid token found at {token_path}. "
        "Run `python auth.py` to complete the local OAuth flow."
    )


def _save_token(creds: Credentials, token_path: Path) -> None:
    token_path.write_text(creds.to_json(), encoding="utf-8")
    logger.info("Saved refreshed credentials to %s", token_path)


def authorize(
    *,
    credentials_path: Path = CREDENTIALS_FILE,
    token_path: Path = TOKEN_FILE,
    port: int = 8080,
) -> Credentials:
    """
    Perform the local OAuth consent flow and persist token.json.

    Uses credentials.run_local_server to open a browser window for manual approval.
    """
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Missing OAuth client secrets at {credentials_path}. "
            "Download your Desktop Client credentials from Google Cloud Console."
        )

    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), HEALTH_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired token during authorize().")
            creds.refresh(Request())
            _save_token(creds, token_path)
            return creds
        if creds and creds.valid:
            logger.info("Existing token.json is still valid; skipping browser flow.")
            return creds

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path),
        scopes=HEALTH_SCOPES,
    )
    creds = flow.run_local_server(port=port, prompt="consent")
    _save_token(creds, token_path)
    logger.info("Authorization complete. token.json is ready for the backend.")
    return creds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    authorize()
