"""
Backward-compatible shim — prefer integrations.llm.create_llm_provider().
"""

from __future__ import annotations

from .llm.gemini import (
    DEFAULT_API_KEY as GEMINI_API_KEY,
    DEFAULT_BACKOFF as GEMINI_RATE_LIMIT_BACKOFF_SECONDS,
    DEFAULT_CALL_DELAY as GEMINI_CALL_DELAY_SECONDS,
    DEFAULT_MAX_RETRIES as GEMINI_RATE_LIMIT_MAX_RETRIES,
    DEFAULT_MODEL as GEMINI_MODEL,
    RATE_LIMIT_USER_REPLY,
    GeminiProvider,
)

# Legacy alias used by older imports/tests.
GeminiClient = GeminiProvider

__all__ = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_CALL_DELAY_SECONDS",
    "GEMINI_RATE_LIMIT_MAX_RETRIES",
    "GEMINI_RATE_LIMIT_BACKOFF_SECONDS",
    "RATE_LIMIT_USER_REPLY",
    "GeminiClient",
    "GeminiProvider",
]
