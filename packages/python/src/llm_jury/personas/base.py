from __future__ import annotations

from dataclasses import dataclass, field

from llm_jury._defaults import DEFAULT_MODEL


@dataclass(slots=True)
class Persona:
    name: str
    role: str
    system_prompt: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    known_bias: str | None = None


@dataclass(slots=True)
class PersonaResponse:
    persona_name: str
    label: str
    confidence: float
    reasoning: str
    key_factors: list[str] = field(default_factory=list)
    dissent_notes: str | None = None
    raw_response: str | None = None
    tokens_used: int = 0
    cost_usd: float | None = None
