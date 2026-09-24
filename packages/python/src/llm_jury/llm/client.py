from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Models whose API does not accept a ``temperature`` parameter.
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "gpt-5")

# Upper bound on how long a provider's Retry-After header can make us wait.
_MAX_RETRY_AFTER_SECONDS = 60.0

_exponential_backoff = wait_exponential(multiplier=1, min=1, max=10)

# Indirection so tests can skip real backoff sleeps.
_sleep = asyncio.sleep

# litellm raises typed errors; we match by class name so we don't take
# a hard runtime dependency on `import litellm` at module load time.
_LITELLM_RETRYABLE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    }
)


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if type(exc).__name__ in _LITELLM_RETRYABLE_NAMES:
        return True
    for attr in ("status_code", "http_status", "status"):
        status = getattr(exc, attr, None)
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            return True
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Delay requested by a ``Retry-After`` header on ``exc``, capped at 60 s.

    Looks at ``exc.response.headers`` then ``exc.headers``. Accepts delta
    seconds or an HTTP date. Returns ``None`` when there is no usable header.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
        if value is None:
            value = headers.get("Retry-After")
    except Exception:
        return None
    if value is None:
        return None

    text = str(value).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return min(max(0.0, seconds), _MAX_RETRY_AFTER_SECONDS)


def _wait_before_retry(retry_state: RetryCallState) -> float:
    """Honour the provider's Retry-After when present, else exponential backoff."""
    outcome = retry_state.outcome
    if outcome is not None and outcome.failed:
        exc = outcome.exception()
        if exc is not None:
            delay = _retry_after_seconds(exc)
            if delay is not None:
                return delay
    return _exponential_backoff(retry_state)


class LLMClient(Protocol):
    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class LiteLLMClient:
    """Production LLM client backed by litellm.

    ``timeout_seconds`` bounds each request (``None`` leaves litellm's own
    default in place). Retryable failures (connection errors, timeouts, 429,
    5xx) are retried up to ``max_attempts`` total attempts, waiting for the
    provider's ``Retry-After`` when it sends one (capped at 60 s) and using
    exponential backoff otherwise. ``api_key`` and ``api_base`` are passed
    to litellm when set.
    """

    def __init__(
        self,
        timeout_seconds: float | None = 60.0,
        max_attempts: int = 3,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, int(max_attempts))
        self.api_key = api_key
        self.api_base = api_base

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrying = AsyncRetrying(
            sleep=_sleep,
            retry=retry_if_exception(_is_retryable_error),
            stop=stop_after_attempt(self.max_attempts),
            wait=_wait_before_retry,
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._complete_once(
                    model, system_prompt, prompt, temperature, response_format
                )
        raise RuntimeError("retry loop ended without a result")  # pragma: no cover

    async def _complete_once(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            from litellm import acompletion, completion_cost  # pragma: no cover
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "litellm is not installed. Install llm-jury with litellm support or inject llm_client."
            ) from exc

        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if _should_send_temperature(model, temperature):
            request["temperature"] = temperature
        if response_format is not None:
            request["response_format"] = response_format
        if self.timeout_seconds is not None:
            request["timeout"] = self.timeout_seconds
        if self.api_key is not None:
            request["api_key"] = self.api_key
        if self.api_base is not None:
            request["api_base"] = self.api_base

        response = await acompletion(**request)
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        cost_usd: float | None = None
        try:
            cost_usd = float(completion_cost(completion_response=response))
        except Exception:
            pass

        return {
            "content": content,
            "tokens": total_tokens,
            "cost_usd": cost_usd,
        }


class NoopLLMClient:
    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("No llm_client configured.")


def _should_send_temperature(model: str, temperature: float | None) -> bool:
    if temperature is None:
        return False
    # Match on the bare model name so provider-prefixed ids such as
    # "openai/gpt-5-mini" are recognised too.
    name = model.lower().rsplit("/", 1)[-1]
    if any(name.startswith(prefix) for prefix in _NO_TEMPERATURE_PREFIXES):
        if temperature != 0.0:
            logger.debug(
                "Model %s does not support temperature; ignoring temperature=%.2f",
                model,
                temperature,
            )
        return False
    return True
