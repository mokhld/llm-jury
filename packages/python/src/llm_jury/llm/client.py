from __future__ import annotations

import logging
from typing import Any, Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Models whose API does not accept a ``temperature`` parameter.
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "gpt-5")


class LLMClient(Protocol):
    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
    ) -> dict[str, Any]: ...


class LiteLLMClient:
    """Production LLM client backed by litellm."""

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
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
    ) -> dict[str, Any]:
        raise RuntimeError("No llm_client configured.")


def _should_send_temperature(model: str, temperature: float | None) -> bool:
    if temperature is None:
        return False
    lower = model.lower()
    if any(lower.startswith(prefix) for prefix in _NO_TEMPERATURE_PREFIXES):
        if temperature != 0.0:
            logger.debug(
                "Model %s does not support temperature; ignoring temperature=%.2f",
                model,
                temperature,
            )
        return False
    return True
