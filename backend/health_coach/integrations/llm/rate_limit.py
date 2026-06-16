"""Shared rate-limit spacing and retry logic for LLM providers."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitedLLMProvider(ABC):
    """Base class with pre/post call spacing and exponential backoff on 429."""

    _last_call_at: float = 0.0
    _call_lock = threading.Lock()

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        call_delay_seconds: float,
        rate_limit_max_retries: int,
        rate_limit_backoff_seconds: float,
        rate_limit_user_reply: str,
    ):
        self._provider_name = provider_name
        self._model_name = model_name
        self._call_delay_seconds = call_delay_seconds
        self._rate_limit_max_retries = rate_limit_max_retries
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._rate_limit_user_reply = rate_limit_user_reply

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def rate_limit_user_reply(self) -> str:
        return self._rate_limit_user_reply

    @staticmethod
    @abstractmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        raise NotImplementedError

    def _wait_for_call_slot(self) -> None:
        if self._call_delay_seconds <= 0:
            return
        with self._call_lock:
            elapsed = time.monotonic() - RateLimitedLLMProvider._last_call_at
            remaining = self._call_delay_seconds - elapsed
            if remaining > 0:
                logger.info(
                    "Waiting %.1fs before %s call (rate spacing).",
                    remaining,
                    self._provider_name,
                )
                time.sleep(remaining)

    def _mark_call_completed(self) -> None:
        with self._call_lock:
            RateLimitedLLMProvider._last_call_at = time.monotonic()

    def _throttle_after_call(self) -> None:
        if self._call_delay_seconds <= 0:
            return
        logger.info(
            "Throttling %.1fs after %s call.",
            self._call_delay_seconds,
            self._provider_name,
        )
        time.sleep(self._call_delay_seconds)
        self._mark_call_completed()

    def _rate_limited_call(self, purpose: str, call: Callable[[], T]) -> T:
        self._wait_for_call_slot()
        last_exc: BaseException | None = None
        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                result = call()
                self._throttle_after_call()
                return result
            except Exception as exc:
                last_exc = exc
                if self.is_rate_limit_error(exc) and attempt < self._rate_limit_max_retries:
                    wait = self._rate_limit_backoff_seconds * (2**attempt)
                    logger.warning(
                        "%s rate limit during %s (attempt %d/%d); retrying in %.1fs.",
                        self._provider_name,
                        purpose,
                        attempt + 1,
                        self._rate_limit_max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    @abstractmethod
    def _complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        images: list[tuple[bytes, str]] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def generate_json(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        images: list[tuple[bytes, str]] | None = None,
    ) -> dict[str, Any]:
        return self._rate_limited_call(
            purpose,
            lambda: self._complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=images,
            ),
        )

    def generate_structured(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.2,
        images: list[tuple[bytes, str]] | None = None,
    ) -> BaseModel | None:
        import json

        try:
            parsed = self.generate_json(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                images=images,
            )
            return response_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.exception(
                "Failed to parse %s structured response for %s: %s",
                self._provider_name,
                purpose,
                exc,
            )
            return None
