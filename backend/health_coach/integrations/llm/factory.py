"""Construct the configured LLM provider."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from .protocol import LLMProvider

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "mistral", "glm")
ROUTING_MODES = ("all_google", "gemini_glm", "gemini_mistral")


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

    if name == "glm":
        from .glm import GLMProvider

        return GLMProvider(**kwargs)

    raise ValueError(
        f"Unknown LLM provider {name!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def _build_glm_text_provider(
    *,
    call_delay_seconds: float | None = None,
    rate_limit_max_retries: int | None = None,
    rate_limit_backoff_seconds: float | None = None,
) -> LLMProvider:
    from .glm import GLMProvider
    from .glm_guard import CACHE_ENABLED, GuardedGLMProvider

    glm = GLMProvider(
        call_delay_seconds=call_delay_seconds,
        rate_limit_max_retries=rate_limit_max_retries,
        rate_limit_backoff_seconds=rate_limit_backoff_seconds,
    )
    if CACHE_ENABLED:
        return GuardedGLMProvider(glm)
    return glm


def _maybe_wrap_fallback(
    primary: LLMProvider,
    *,
    enable_fallback: bool | None = None,
) -> LLMProvider:
    if enable_fallback is False:
        return primary

    fallback_name = (os.getenv("LLM_FALLBACK_PROVIDER") or "").strip().lower()
    if enable_fallback is not True and not fallback_name:
        return primary
    if not fallback_name:
        if primary.provider_name == "glm":
            fallback_name = "gemini"
        elif primary.provider_name == "mistral":
            fallback_name = "gemini"
        else:
            fallback_name = "mistral" if primary.provider_name == "gemini" else "gemini"
    if fallback_name == primary.provider_name:
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
        primary.provider_name,
        fallback_name,
    )
    return FallbackLLMProvider(primary=primary, fallback=fallback)


def _build_dual_model_provider(
    *,
    text_provider_name: str,
    api_key: str | None = None,
    model_name: str | None = None,
    call_delay_seconds: float | None = None,
    rate_limit_max_retries: int | None = None,
    rate_limit_backoff_seconds: float | None = None,
    enable_fallback: bool | None = None,
) -> LLMProvider:
    from .routing import DualModelLLMProvider

    vision = _build_single_provider(
        "gemini",
        api_key=api_key,
        model_name=model_name,
        call_delay_seconds=call_delay_seconds,
        rate_limit_max_retries=rate_limit_max_retries,
        rate_limit_backoff_seconds=rate_limit_backoff_seconds,
    )

    if text_provider_name == "glm":
        text = _build_glm_text_provider(
            call_delay_seconds=call_delay_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        )
    else:
        text = _build_single_provider(
            text_provider_name,
            call_delay_seconds=call_delay_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        )

    text = _maybe_wrap_fallback(text, enable_fallback=enable_fallback)

    logger.info(
        "Dual-model routing enabled: vision=gemini, text=%s",
        text.provider_name,
    )
    return DualModelLLMProvider(vision=vision, text=text)


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
      LLM_ROUTING_MODE=all_google|gemini_glm|gemini_mistral  (default: all_google)
      LLM_PROVIDER=gemini|mistral|glm  (default: gemini; used in all_google mode)
      LLM_FALLBACK_PROVIDER=mistral  (optional; auto-failover when primary errors)
      LLM_MODEL, LLM_API_KEY       (optional generic overrides)
      Provider-specific keys still work (GEMINI_*, MISTRAL_*, CLOUDFLARE_*).
    """
    routing_mode = (os.getenv("LLM_ROUTING_MODE") or "all_google").strip().lower()

    if routing_mode == "gemini_glm":
        return _build_dual_model_provider(
            text_provider_name="glm",
            api_key=api_key,
            model_name=model_name,
            call_delay_seconds=call_delay_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
            enable_fallback=enable_fallback,
        )

    if routing_mode == "gemini_mistral":
        return _build_dual_model_provider(
            text_provider_name="mistral",
            api_key=api_key,
            model_name=model_name,
            call_delay_seconds=call_delay_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
            enable_fallback=enable_fallback,
        )

    if routing_mode != "all_google":
        raise ValueError(
            f"Unknown LLM_ROUTING_MODE {routing_mode!r}. "
            f"Supported: {', '.join(ROUTING_MODES)}."
        )

    primary_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    generic_model = model_name or os.getenv("LLM_MODEL") or None

    if primary_name == "glm":
        primary = _build_glm_text_provider(
            call_delay_seconds=call_delay_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        )
    else:
        primary = _build_single_provider(
            primary_name,
            api_key=api_key,
            model_name=generic_model,
            call_delay_seconds=call_delay_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        )

    return _maybe_wrap_fallback(primary, enable_fallback=enable_fallback)
