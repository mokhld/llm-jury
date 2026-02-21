from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeLLMReply:
    content: str
    tokens: int = 10
    cost_usd: float = 0.001


class FakeLLMClient:
    def __init__(self, responses: dict[str, FakeLLMReply] | None = None):
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "temperature": temperature,
            }
        )

        if model in self.responses:
            reply = self.responses[model]
            return {
                "content": reply.content,
                "tokens": reply.tokens,
                "cost_usd": reply.cost_usd,
            }

        persona_name = None
        for key in self.responses:
            if key in prompt or key in system_prompt:
                persona_name = key
                break

        if persona_name is None and self.responses:
            persona_name = next(iter(self.responses.keys()))

        if persona_name and persona_name in self.responses:
            reply = self.responses[persona_name]
            return {
                "content": reply.content,
                "tokens": reply.tokens,
                "cost_usd": reply.cost_usd,
            }

        default = {
            "label": "safe",
            "confidence": 0.75,
            "reasoning": "Default safe classification.",
            "key_factors": ["default"],
        }
        return {
            "content": json.dumps(default),
            "tokens": 10,
            "cost_usd": 0.001,
        }
