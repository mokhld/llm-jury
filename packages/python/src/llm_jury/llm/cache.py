from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

from .client import LLMClient


def _cache_key(
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float | None,
    response_format: dict[str, Any] | None,
) -> tuple[str, str, str, str, str]:
    rf = json.dumps(response_format, sort_keys=True) if response_format else ""
    temp = "" if temperature is None else f"{temperature:.6f}"
    return (model, system_prompt, prompt, temp, rf)


class CachingLLMClient:
    """In-memory LRU cache wrapper for any `LLMClient`.

    Opt-in. Wrap your client explicitly:

        jury = Jury(..., llm_client=CachingLLMClient(LiteLLMClient()))

    Key: `(model, system_prompt, prompt, temperature, response_format)`.
    Only successful responses are cached; exceptions propagate without
    being stored. The cache is in-process and not shared across `Jury`
    instances unless you share the `CachingLLMClient` instance.

    Caches at non-zero temperature too. If you need fresh stochastic
    samples per call, don't wrap — the cache has no knowledge of
    whether a model is deterministic at the chosen temperature.
    """

    def __init__(
        self,
        inner: LLMClient,
        max_size: int = 1000,
        ttl_seconds: float | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive or None")
        self._inner = inner
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[
            tuple[str, str, str, str, str], tuple[float, dict[str, Any]]
        ] = OrderedDict()
        self.hits = 0
        self.misses = 0

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = _cache_key(model, system_prompt, prompt, temperature, response_format)
        entry = self._cache.get(key)
        if entry is not None:
            stored_at, value = entry
            if (
                self._ttl_seconds is None
                or time.monotonic() - stored_at < self._ttl_seconds
            ):
                self._cache.move_to_end(key)
                self.hits += 1
                return value
            del self._cache[key]

        value = await self._inner.complete(
            model,
            system_prompt,
            prompt,
            temperature,
            response_format,
        )
        self.misses += 1
        self._cache[key] = (time.monotonic(), value)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return value

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0
