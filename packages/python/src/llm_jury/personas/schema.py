"""JSON Schemas for persona, classifier and judge responses.

The schemas are sent to the LLM as OpenAI-style ``response_format`` so the
provider itself enforces well-formed output. LiteLLM forwards the schema
to any provider that supports structured output (OpenAI, Anthropic via
tool-use, etc.) and falls back to prompt-only enforcement otherwise.

Strict-mode JSON Schema requires ``additionalProperties: false`` and every
property listed in ``required`` — optional fields like ``dissent_notes`` are
modeled as nullable strings instead of being omitted.
"""

from __future__ import annotations

from typing import Any


def _label_property(labels: list[str]) -> dict[str, Any]:
    label_property: dict[str, Any] = {"type": "string"}
    if labels:
        label_property["enum"] = list(labels)
    return label_property


def _strict_response_format(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def build_persona_response_schema(labels: list[str]) -> dict[str, Any]:
    """Return a ``response_format`` payload constraining persona output.

    The returned dict matches OpenAI's JSON Schema response format:
    ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": true}}``.
    """
    return _strict_response_format(
        "persona_response",
        {
            "label": _label_property(labels),
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
            "key_factors": {"type": "array", "items": {"type": "string"}},
            "dissent_notes": {"type": ["string", "null"]},
        },
    )


def build_classifier_response_schema(labels: list[str]) -> dict[str, Any]:
    """Return a ``response_format`` payload constraining ``LLMClassifier`` output
    to ``{"label": <one of labels>, "confidence": <number>}``."""
    return _strict_response_format(
        "classifier_response",
        {
            "label": _label_property(labels),
            "confidence": {"type": "number"},
        },
    )


def build_judge_response_schema(labels: list[str]) -> dict[str, Any]:
    """Return a ``response_format`` payload constraining ``LLMJudge`` output."""
    return _strict_response_format(
        "judge_response",
        {
            "label": _label_property(labels),
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
            "key_agreements": {"type": "array", "items": {"type": "string"}},
            "key_disagreements": {"type": "array", "items": {"type": "string"}},
            "decisive_factor": {"type": "string"},
        },
    )
