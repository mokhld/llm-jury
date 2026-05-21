"""JSON Schema for persona responses.

The schema is sent to the LLM as OpenAI-style ``response_format`` so the
provider itself enforces well-formed output. LiteLLM forwards the schema
to any provider that supports structured output (OpenAI, Anthropic via
tool-use, etc.) and falls back to prompt-only enforcement otherwise.

Strict-mode JSON Schema requires ``additionalProperties: false`` and every
property listed in ``required`` — optional fields like ``dissent_notes`` are
modeled as nullable strings instead of being omitted.
"""
from __future__ import annotations

from typing import Any


def build_persona_response_schema(labels: list[str]) -> dict[str, Any]:
    """Return a ``response_format`` payload constraining persona output.

    The returned dict matches OpenAI's JSON Schema response format:
    ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": true}}``.
    """
    label_property: dict[str, Any] = {"type": "string"}
    if labels:
        label_property["enum"] = list(labels)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "label": label_property,
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
            "key_factors": {"type": "array", "items": {"type": "string"}},
            "dissent_notes": {"type": ["string", "null"]},
        },
        "required": ["label", "confidence", "reasoning", "key_factors", "dissent_notes"],
        "additionalProperties": False,
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "persona_response",
            "schema": schema,
            "strict": True,
        },
    }
