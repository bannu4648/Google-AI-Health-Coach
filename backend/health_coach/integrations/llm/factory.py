"""Construct the configured LLM provider."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from .protocol import LLMProvider

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "mistral")


def _build_single_provider(
    name: str,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    call_delay_seconds: float | None = None,
    rate_limit_max_retries: int | None = None,
    rate_limit_backoff_seconds: float | None = None,
) -> LLMProvider:
    kwargs: dict = {
        "api_key": api_key,
        "call_delay_seconds": call_delay_seconds,
        "rate_limit_max_retries": rate_limit_max_retries,
        "rate_limit_backoff_seconds": rate_limit_backoff_seconds,
    }
    if model_name:
        kwargs["model_name"] = model_name

    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(**kwargs)

    if name == "mistral":
        from .mistral import MistralProvider

        return MistralProvider(**kwargs)

    raise ValueError(
        f"Unknown LLM provider {name!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def create_llm_provider(
    provider: str | None = None,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    call_delay_seconds: float | None = None,
    rate_limit_max_retries: int | None = None,
    rate_limit_backoff_seconds: float | None = None,
    enable_fallback: bool | None = None,
) -> LLMProvider:
    """
    Build an LLM provider from env or explicit args.

    Env:
      LLM_PROVIDER=gemini|mistral  (default: gemini)
      LLM_FALLBACK_PROVIDER=mistral  (optional; auto-failover when primary errors)
      LLM_MODEL, LLM_API_KEY       (optional generic overrides)
      Provider-specific keys still work (GEMINI_*, MISTRAL_*).
    """
    primary_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    generic_model = model_name or os.getenv("LLM_MODEL") or None

    primary = _build_single_provider(
        primary_name,
        api_key=api_key,
        model_name=generic_model,
        call_delay_seconds=call_delay_seconds,
        rate_limit_max_retries=rate_limit_max_retries,
        rate_limit_backoff_seconds=rate_limit_backoff_seconds,
    )

    if enable_fallback is False:
        return primary

    fallback_name = (os.getenv("LLM_FALLBACK_PROVIDER") or "").strip().lower()
    if enable_fallback is not True and not fallback_name:
        return primary
    if not fallback_name:
        fallback_name = "mistral" if primary_name == "gemini" else "gemini"
    if fallback_name == primary_name:
        return primary

    try:
        fallback = _build_single_provider(
            fallback_name,
            model_name=None,
            api_key=None,
            call_delay_seconds=None,
            rate_limit_max_retries=None,
            rate_limit_backoff_seconds=None,
        )
    except (ValueError, ImportError) as exc:
        logger.warning("LLM fallback provider %s unavailable: %s", fallback_name, exc)
        return primary

    from .fallback import FallbackLLMProvider

    logger.info(
        "LLM failover enabled: primary=%s, fallback=%s",
        primary_name,
        fallback_name,
    )
    return FallbackLLMProvider(primary=primary, fallback=fallback)
