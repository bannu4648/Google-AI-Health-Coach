"""Mobile-friendly Google OAuth routes (via PUBLIC_BASE_URL / ngrok)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from ..integrations.google_auth import (
    GoogleAuthRequiredError,
    build_google_authorization_url,
    complete_mobile_auth,
    get_oauth_pending_state,
    oauth_callback_uri,
)
from ..integrations.whatsapp import send_text_message

logger = logging.getLogger(__name__)

router = APIRouter(tags=["google-oauth"])


@router.get("/auth/google/start")
async def google_auth_start(state: str = Query(..., min_length=8)) -> RedirectResponse:
    """Redirect phone browser to Google sign-in (link sent via WhatsApp)."""
    try:
        auth_url = build_google_authorization_url(state=state)
    except GoogleAuthRequiredError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/auth/google/callback")
async def google_auth_callback(
    code: str = Query(...),
    state: str = Query(...),
) -> HTMLResponse:
    """OAuth redirect target — exchanges code for token.json and notifies user on WhatsApp."""
    pending = get_oauth_pending_state(state)
    phone = (pending or {}).get("phone")
    try:
        complete_mobile_auth(state=state, code=code)
    except GoogleAuthRequiredError as exc:
        logger.warning("Mobile OAuth callback failed: %s", exc)
        return HTMLResponse(
            content=_html_page(
                title="Sign-in failed",
                body=f"<p>{exc}</p><p>Message your coach on WhatsApp to get a new link.</p>",
                success=False,
            ),
            status_code=400,
        )
    except Exception as exc:
        logger.exception("Mobile OAuth callback error: %s", exc)
        return HTMLResponse(
            content=_html_page(
                title="Sign-in failed",
                body="<p>Something went wrong saving your Google Health connection.</p>",
                success=False,
            ),
            status_code=500,
        )

    if phone:
        send_text_message(
            phone,
            "Google Health is connected again. You can message me your question now.",
        )

    return HTMLResponse(
        content=_html_page(
            title="Connected",
            body=(
                "<p>Google Health is connected.</p>"
                "<p>Return to WhatsApp and send your message again.</p>"
            ),
            success=True,
        )
    )


@router.get("/auth/google/status")
async def google_auth_status() -> dict[str, str | bool | None]:
    """Quick check that mobile OAuth is configured."""
    callback = oauth_callback_uri()
    return {
        "mobile_oauth_configured": bool(callback),
        "callback_uri": callback,
    }


def _html_page(*, title: str, body: str, success: bool) -> str:
    color = "#137333" if success else "#c5221f"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title></head>
<body style="font-family:system-ui,sans-serif;padding:2rem;max-width:32rem;margin:auto;">
<h1 style="color:{color}">{title}</h1>
{body}
</body></html>"""
