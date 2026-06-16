"""Construct the configured LLM provider."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .protocol import LLMProvider

load_dotenv()

SUPPORTED_PROVIDERS = ("gemini", "mistral")


def create_llm_provider(
    provider: str | None = None,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    call_delay_seconds: float | None = None,
    rate_limit_max_retries: int | None = None,
    rate_limit_backoff_seconds: float | None = None,
) -> LLMProvider:
    """
    Build an LLM provider from env or explicit args.

    Env:
      LLM_PROVIDER=gemini|mistral  (default: gemini)
      LLM_MODEL, LLM_API_KEY       (optional generic overrides)
      Provider-specific keys still work (GEMINI_*, MISTRAL_*).
    """
    name = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    kwargs = {
        "api_key": api_key,
        "model_name": model_name or os.getenv("LLM_MODEL") or None,
        "call_delay_seconds": call_delay_seconds,
        "rate_limit_max_retries": rate_limit_max_retries,
        "rate_limit_backoff_seconds": rate_limit_backoff_seconds,
    }
    # Drop None model_name so provider defaults apply
    if kwargs["model_name"] is None:
        del kwargs["model_name"]

    if name == "gemini":
        from .gemini import GeminiProvider as _Gemini

        return _Gemini(**kwargs)

    if name == "mistral":
        from .mistral import MistralProvider

        return MistralProvider(**kwargs)

    raise ValueError(
        f"Unknown LLM_PROVIDER {name!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}. "
        "OpenAI support can be added by implementing OpenAIProvider in integrations/llm/."
    )
