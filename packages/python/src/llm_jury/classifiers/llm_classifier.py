from __future__ import annotations

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.llm.client import LiteLLMClient, LLMClient
from llm_jury.personas.schema import build_classifier_response_schema
from llm_jury.utils import (
    match_label,
    parse_confidence,
    payload_cost,
    safe_json_parse,
    strip_markdown_fences,
    wrap_untrusted,
)

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
        cleaned = [str(label).strip() for label in (labels or []) if str(label).strip()]
        if not cleaned:
            raise ValueError(
                "LLMClassifier requires at least one non-empty label. "
                "Pass labels=['safe', 'unsafe'] (or similar) when constructing."
            )
        self.model = model
        self.labels = cleaned
        self.system_prompt = (
            system_prompt
            or "Classify the text and return JSON with label and confidence."
        )
        self.llm_client = llm_client or LiteLLMClient()
        self.temperature = temperature

    async def classify(self, text: str) -> ClassificationResult:
        """Classify ``text`` with one LLM call.

        Unusable output never raises: invalid JSON, a label outside
        ``labels`` or an invalid confidence returns confidence 0.0 (so the
        Jury escalates) with the reason in ``raw_output["error"]``.
        """
        prompt = (
            "Classify the following text into one of the available labels.\n"
            f"Labels: {', '.join(self.labels) if self.labels else 'any'}\n"
            f"Text:\n{wrap_untrusted(text)}\n"
            'Respond with JSON: {"label":"...","confidence":0.0-1.0}.'
        )
        payload = await self.llm_client.complete(
            model=self.model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            temperature=self.temperature,
            response_format=build_classifier_response_schema(self.labels),
        )
        raw_content = str(payload.get("content") or "")
        call_cost = payload_cost(payload)
        data = safe_json_parse(strip_markdown_fences(raw_content))
        fallback_label = self.labels[0] if self.labels else "unknown"
        if data is None:
            return ClassificationResult(
                label=fallback_label,
                confidence=0.0,
                raw_output={"raw_content": raw_content, "error": "invalid_json"},
                cost_usd=call_cost,
            )

        label = match_label(data.get("label"), self.labels)
        if label is None:
            return ClassificationResult(
                label=fallback_label,
                confidence=0.0,
                raw_output={"raw_content": raw_content, "error": "label_not_in_labels"},
                cost_usd=call_cost,
            )

        confidence = parse_confidence(data.get("confidence"))
        if confidence is None:
            return ClassificationResult(
                label=label,
                confidence=0.0,
                raw_output={"raw_content": raw_content, "error": "invalid_confidence"},
                cost_usd=call_cost,
            )

        return ClassificationResult(
            label=label,
            confidence=confidence,
            raw_output=data,
            cost_usd=call_cost,
        )
