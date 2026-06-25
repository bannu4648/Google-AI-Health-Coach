"""GLM-5.2 LLM provider via Cloudflare Workers AI."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

from .rate_limit import RateLimitedLLMProvider

load_dotenv()

DEFAULT_MODEL = os.getenv("GLM_MODEL", "@cf/zai-org/glm-5.2")
DEFAULT_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
DEFAULT_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    os.getenv("CLOUDFLARE_AUTH_TOKEN", ""),
)
DEFAULT_CALL_DELAY = float(
    os.getenv("GLM_CALL_DELAY_SECONDS", os.getenv("LLM_CALL_DELAY_SECONDS", "0"))
)
DEFAULT_MAX_RETRIES = int(
    os.getenv("GLM_RATE_LIMIT_MAX_RETRIES", os.getenv("LLM_RATE_LIMIT_MAX_RETRIES", "3"))
)
DEFAULT_BACKOFF = float(
    os.getenv(
        "GLM_RATE_LIMIT_BACKOFF_SECONDS",
        os.getenv("LLM_RATE_LIMIT_BACKOFF_SECONDS", "2"),
    )
)

JSON_SUFFIX = "\n\nRespond with a single valid JSON object only. No markdown fences."

RATE_LIMIT_USER_REPLY = (
    "The Cloudflare Workers AI rate limit was hit — I've queued a short pause and will be "
    "ready again in a few seconds. Please resend your message."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _run_url(account_id: str, model: str) -> str:
    return (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{model}"
    )


def _extract_response_text(payload: dict[str, Any]) -> str:
    """Parse Cloudflare /run response shapes defensively."""
    if not payload:
        return "{}"

    if payload.get("success") is False:
        errors = payload.get("errors") or []
        message = "; ".join(str(e) for e in errors) or "Cloudflare Workers AI request failed"
        raise RuntimeError(message)

    result = payload.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("response"), str):
            return result["response"]
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content

    if isinstance(payload.get("response"), str):
        return payload["response"]

    return json.dumps(result if result is not None else payload)


def _parse_json_text(raw: str) -> dict[str, Any]:
    cleaned = _FENCE_RE.sub("", raw.strip())
    return json.loads(cleaned or "{}")


class GLMProvider(RateLimitedLLMProvider):
    """GLM-5.2 JSON provider via Cloudflare Workers AI. Vision is not supported."""

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        model_name: str = DEFAULT_MODEL,
        call_delay_seconds: float | None = None,
        rate_limit_max_retries: int | None = None,
        rate_limit_backoff_seconds: float | None = None,
    ):
        token = api_key if api_key is not None else DEFAULT_API_TOKEN
        acct = account_id if account_id is not None else DEFAULT_ACCOUNT_ID
        if not acct:
            raise ValueError("Set CLOUDFLARE_ACCOUNT_ID in your .env file.")
        if not token:
            raise ValueError("Set CLOUDFLARE_API_TOKEN in your .env file.")

        self._account_id = acct
        self._api_token = token
        super().__init__(
            provider_name="glm",
            model_name=model_name,
            call_delay_seconds=DEFAULT_CALL_DELAY if call_delay_seconds is None else call_delay_seconds,
            rate_limit_max_retries=DEFAULT_MAX_RETRIES
            if rate_limit_max_retries is None
            else rate_limit_max_retries,
            rate_limit_backoff_seconds=DEFAULT_BACKOFF
            if rate_limit_backoff_seconds is None
            else rate_limit_backoff_seconds,
            rate_limit_user_reply=RATE_LIMIT_USER_REPLY,
        )

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        if "429" in message or "rate limit" in message or "quota" in message:
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        return False

    def _complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        images: list[tuple[bytes, str]] | None,
    ) -> dict[str, Any]:
        if images:
            raise NotImplementedError(
                "GLM provider does not support vision in this project. "
                "Set LLM_ROUTING_MODE=gemini_glm for food photos via Gemini."
            )

        url = _run_url(self._account_id, self._model_name)
        body = {
            "messages": [
                {"role": "system", "content": f"{system_prompt}{JSON_SUFFIX}"},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        if response.status_code == 429:
            raise RuntimeError(f"429 rate limit exceeded: {response.text[:200]}")
        response.raise_for_status()
        raw = _extract_response_text(response.json())
        return _parse_json_text(raw)
