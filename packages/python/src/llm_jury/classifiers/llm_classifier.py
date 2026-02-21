from __future__ import annotations

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.llm.client import LLMClient, LiteLLMClient
from llm_jury.utils import clamp_confidence, safe_json_parse, strip_markdown_fences

from .base import ClassificationResult, Classifier


class LLMClassifier(Classifier):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        labels: list[str] | None = None,
        system_prompt: str | None = None,
        llm_client: LLMClient | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.labels = labels or []
        self.system_prompt = system_prompt or "Classify the text and return JSON with label and confidence."
        self.llm_client = llm_client or LiteLLMClient()
        self.temperature = temperature

    async def classify(self, text: str) -> ClassificationResult:
        prompt = (
            "Classify the following text into one of the available labels.\n"
            f"Labels: {', '.join(self.labels) if self.labels else 'any'}\n"
            f"Text: {text}\n"
            "Respond with JSON: {\"label\":\"...\",\"confidence\":0.0-1.0}."
        )
        payload = await self.llm_client.complete(
            model=self.model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            temperature=self.temperature,
        )
        raw_content = str(payload.get("content", "{}"))
        call_cost = payload.get("cost_usd")
        data = safe_json_parse(strip_markdown_fences(raw_content))
        fallback_label = self.labels[0] if self.labels else "unknown"
        if data is None:
            return ClassificationResult(
                label=fallback_label,
                confidence=0.0,
                raw_output={"raw_content": raw_content, "error": "invalid_json"},
                cost_usd=call_cost,
            )
        return ClassificationResult(
            label=str(data.get("label", fallback_label)),
            confidence=clamp_confidence(data.get("confidence", 0.0)),
            raw_output=data,
            cost_usd=call_cost,
        )
