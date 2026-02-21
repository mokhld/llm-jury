from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.classifiers.base import ClassificationResult
from llm_jury.llm.client import LLMClient, LiteLLMClient
from llm_jury.personas.base import Persona, PersonaResponse
from llm_jury.utils import clamp_confidence, safe_json_parse, strip_markdown_fences

logger = logging.getLogger(__name__)

_SUMMARISATION_PROMPT = (
    "You are a neutral summarisation agent. You have observed a structured debate "
    "among domain experts about classifying a piece of text.\n\n"
    "Produce a concise synthesis that covers:\n"
    "1. The main arguments from each side\n"
    "2. Points of consensus among the experts\n"
    "3. Unresolved disagreements\n\n"
    "Be factual and impartial. Do not add your own classification."
)

_DELIBERATION_INSTRUCTIONS = (
    "You have seen the initial assessments from all experts on this input. "
    "You MUST:\n"
    "(i) Engage with at least one other expert's reasoning — agree or disagree "
    "with supporting rationale.\n"
    "(ii) Revise your own classification if you find their counter-arguments compelling.\n"
    "(iii) Re-evaluate the input through the interpretive lens of at least one other expert's "
    "perspective, considering both intent and impact.\n\n"
    "Then provide your revised assessment."
)


class DebateMode(str, Enum):
    INDEPENDENT = "independent"
    SEQUENTIAL = "sequential"
    DELIBERATION = "deliberation"
    ADVERSARIAL = "adversarial"


@dataclass(slots=True)
class DebateConfig:
    mode: DebateMode = DebateMode.DELIBERATION
    max_rounds: int = 2
    include_primary_result: bool = True
    include_confidence: bool = True


@dataclass(slots=True)
class DebateTranscript:
    input_text: str
    primary_result: ClassificationResult
    rounds: list[list[PersonaResponse]]
    duration_ms: int
    total_tokens: int
    total_cost_usd: float | None
    summary: str | None = None


class DebateEngine:
    def __init__(
        self,
        personas: list[Persona],
        config: DebateConfig | None = None,
        llm_client: LLMClient | None = None,
        concurrency: int = 5,
    ) -> None:
        self.personas = personas
        self.config = config or DebateConfig()
        self.llm_client = llm_client or LiteLLMClient()
        self.concurrency = max(1, concurrency)

    async def debate(
        self,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        max_cost_usd: float | None = None,
    ) -> DebateTranscript:
        start = time.perf_counter()
        rounds: list[list[PersonaResponse]] = []
        total_tokens = 0
        total_cost = 0.0
        summary: str | None = None

        if not self.personas:
            return DebateTranscript(
                input_text=text,
                primary_result=primary_result,
                rounds=[],
                duration_ms=int((time.perf_counter() - start) * 1000),
                total_tokens=0,
                total_cost_usd=0.0,
            )

        if self.config.mode in (DebateMode.INDEPENDENT, DebateMode.ADVERSARIAL):
            responses = await self._run_round(text, primary_result, labels, prior_rounds=[])
            rounds.append(responses)
            for response in responses:
                total_tokens += response.tokens_used
                total_cost += float(response.cost_usd or 0.0)

        elif self.config.mode == DebateMode.SEQUENTIAL:
            responses: list[PersonaResponse] = []
            for persona in self.personas:
                response = await self._query_persona(
                    persona=persona,
                    text=text,
                    primary_result=primary_result,
                    labels=labels,
                    prior_rounds=[responses] if responses else [],
                )
                responses.append(response)
                total_tokens += response.tokens_used
                total_cost += float(response.cost_usd or 0.0)
                if max_cost_usd is not None and total_cost > max_cost_usd:
                    break
            rounds.append(responses)

        elif self.config.mode == DebateMode.DELIBERATION:
            # Stage 1: Initial opinions (parallel, independent)
            first_round = await self._run_round(text, primary_result, labels, prior_rounds=[])
            rounds.append(first_round)
            for response in first_round:
                total_tokens += response.tokens_used
                total_cost += float(response.cost_usd or 0.0)

            if max_cost_usd is not None and total_cost > max_cost_usd:
                return DebateTranscript(
                    input_text=text,
                    primary_result=primary_result,
                    rounds=rounds,
                    duration_ms=int((time.perf_counter() - start) * 1000),
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                )

            # Stage 2: Structured debate rounds (personas engage with prior opinions)
            for _ in range(1, max(1, self.config.max_rounds)):
                current = await self._run_deliberation_round(
                    text, primary_result, labels, prior_rounds=rounds,
                )
                rounds.append(current)
                for response in current:
                    total_tokens += response.tokens_used
                    total_cost += float(response.cost_usd or 0.0)

                if max_cost_usd is not None and total_cost > max_cost_usd:
                    break
                if self._consensus_reached(current):
                    break

            # Stage 3: Summarisation
            if not (max_cost_usd is not None and total_cost > max_cost_usd):
                summary, s_tokens, s_cost = await self._summarise(text, labels, rounds)
                total_tokens += s_tokens
                total_cost += s_cost

        duration_ms = int((time.perf_counter() - start) * 1000)
        return DebateTranscript(
            input_text=text,
            primary_result=primary_result,
            rounds=rounds,
            duration_ms=duration_ms,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Round runners
    # ------------------------------------------------------------------

    async def _run_round(
        self,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        prior_rounds: list[list[PersonaResponse]],
    ) -> list[PersonaResponse]:
        sem = asyncio.Semaphore(self.concurrency)

        async def _wrapped(persona: Persona) -> PersonaResponse:
            async with sem:
                return await self._query_persona(persona, text, primary_result, labels, prior_rounds)

        return list(await asyncio.gather(*[_wrapped(persona) for persona in self.personas]))

    async def _run_deliberation_round(
        self,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        prior_rounds: list[list[PersonaResponse]],
    ) -> list[PersonaResponse]:
        sem = asyncio.Semaphore(self.concurrency)

        async def _wrapped(persona: Persona) -> PersonaResponse:
            async with sem:
                return await self._query_persona_deliberation(
                    persona, text, primary_result, labels, prior_rounds,
                )

        return list(await asyncio.gather(*[_wrapped(persona) for persona in self.personas]))

    # ------------------------------------------------------------------
    # Persona query
    # ------------------------------------------------------------------

    async def _query_persona(
        self,
        persona: Persona,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        prior_rounds: list[list[PersonaResponse]],
    ) -> PersonaResponse:
        prompt = self._build_persona_prompt(persona, text, primary_result, labels, prior_rounds)
        return await self._call_persona(persona, prompt, labels)

    async def _query_persona_deliberation(
        self,
        persona: Persona,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        prior_rounds: list[list[PersonaResponse]],
    ) -> PersonaResponse:
        prompt = self._build_deliberation_prompt(persona, text, primary_result, labels, prior_rounds)
        return await self._call_persona(persona, prompt, labels)

    async def _call_persona(
        self,
        persona: Persona,
        prompt: str,
        labels: list[str],
    ) -> PersonaResponse:
        payload = await self.llm_client.complete(
            model=persona.model,
            system_prompt=persona.system_prompt,
            prompt=prompt,
            temperature=persona.temperature,
        )
        raw_content = payload.get("content", "")
        response = self._parse_persona_response(raw_content, persona.name, labels)
        response.raw_response = raw_content
        response.tokens_used = int(payload.get("tokens", 0) or 0)
        response.cost_usd = float(payload.get("cost_usd", 0.0) or 0.0)
        return response

    # ------------------------------------------------------------------
    # Summarisation (Stage 3)
    # ------------------------------------------------------------------

    async def _summarise(
        self,
        text: str,
        labels: list[str],
        rounds: list[list[PersonaResponse]],
    ) -> tuple[str, int, float]:
        """Produce a structured summary of the debate. Returns (summary, tokens, cost)."""
        parts = [
            f"## Input\n\n{text}\n",
            f"## Labels\n\n{', '.join(labels)}\n",
        ]

        for r_idx, round_responses in enumerate(rounds):
            heading = "Initial Expert Opinions" if r_idx == 0 else f"Revised Opinions (Round {r_idx + 1})"
            parts.append(f"## {heading}\n")
            for resp in round_responses:
                parts.append(
                    f"**{resp.persona_name}**: {resp.label} (confidence: {resp.confidence:.2f})\n"
                    f"Reasoning: {resp.reasoning}\n"
                )

        parts.append(
            "\nProduce your synthesis now. Focus on arguments, consensus, and disagreements."
        )

        payload = await self.llm_client.complete(
            model=self.personas[0].model if self.personas else DEFAULT_MODEL,
            system_prompt=_SUMMARISATION_PROMPT,
            prompt="\n".join(parts),
        )
        summary_text = payload.get("content", "")
        tokens = int(payload.get("tokens", 0) or 0)
        cost = float(payload.get("cost_usd", 0.0) or 0.0)
        return summary_text, tokens, cost

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_persona_prompt(
        self,
        persona: Persona,
        text: str,
        primary: ClassificationResult,
        labels: list[str],
        prior_rounds: list[list[PersonaResponse]],
    ) -> str:
        parts = [f"## Persona\n\n{persona.name}: {persona.role}\n", f"## Input to Classify\n\n{text}\n"]
        parts.append(f"## Available Labels\n\n{', '.join(labels)}\n")

        if self.config.mode == DebateMode.ADVERSARIAL:
            persona_index = self.personas.index(persona)
            stance = "Prosecution" if persona_index % 2 == 0 else "Defense"
            parts.append(
                "## Adversarial Role\n\n"
                f"You are assigned the **{stance}** side. "
                "Argue this stance rigorously while remaining truthful to the evidence.\n"
            )

        if self.config.include_primary_result:
            confidence_suffix = (
                f" (confidence: {primary.confidence:.2f})" if self.config.include_confidence else ""
            )
            parts.append(
                "## Primary Classifier Result\n\n"
                f"Label: {primary.label}{confidence_suffix}\n"
                "Note: This was flagged as low-confidence and escalated to you.\n"
            )

        if prior_rounds:
            parts.append("## Previous Assessments\n")
            for idx, round_responses in enumerate(prior_rounds):
                parts.append(f"\n### Round {idx + 1}\n")
                for response in round_responses:
                    parts.append(
                        f"**{response.persona_name}**: {response.label} (confidence: {response.confidence:.2f})\n"
                        f"Reasoning: {response.reasoning}\n"
                    )

        parts.append(self._json_response_block())
        return "\n".join(parts)

    def _build_deliberation_prompt(
        self,
        persona: Persona,
        text: str,
        primary: ClassificationResult,
        labels: list[str],
        prior_rounds: list[list[PersonaResponse]],
    ) -> str:
        parts = [f"## Persona\n\n{persona.name}: {persona.role}\n", f"## Input to Classify\n\n{text}\n"]
        parts.append(f"## Available Labels\n\n{', '.join(labels)}\n")

        if self.config.include_primary_result:
            confidence_suffix = (
                f" (confidence: {primary.confidence:.2f})" if self.config.include_confidence else ""
            )
            parts.append(
                "## Primary Classifier Result\n\n"
                f"Label: {primary.label}{confidence_suffix}\n"
                "Note: This was flagged as low-confidence and escalated to you.\n"
            )

        if prior_rounds:
            parts.append("## Initial Expert Opinions\n")
            for response in prior_rounds[0]:
                parts.append(
                    f"**{response.persona_name}**: {response.label} (confidence: {response.confidence:.2f})\n"
                    f"Reasoning: {response.reasoning}\n"
                )

            for r_idx in range(1, len(prior_rounds)):
                parts.append(f"\n## Revised Opinions (Round {r_idx + 1})\n")
                for response in prior_rounds[r_idx]:
                    parts.append(
                        f"**{response.persona_name}**: {response.label} (confidence: {response.confidence:.2f})\n"
                        f"Reasoning: {response.reasoning}\n"
                    )

        parts.append(f"\n## Deliberation Instructions\n\n{_DELIBERATION_INSTRUCTIONS}\n")
        parts.append(self._json_response_block())
        return "\n".join(parts)

    @staticmethod
    def _json_response_block() -> str:
        return (
            "\n## Your Assessment\n\n"
            "Provide your classification. Respond ONLY with valid JSON:\n"
            "```json\n"
            "{\n"
            '  "label": "<your classification>",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "reasoning": "<your full reasoning>",\n'
            '  "key_factors": ["<factor 1>", "<factor 2>"],\n'
            '  "dissent_notes": "<optional rebuttal against other experts>"\n'
            "}\n"
            "```"
        )

    # ------------------------------------------------------------------
    # Parsing + consensus
    # ------------------------------------------------------------------

    def _parse_persona_response(
        self,
        raw: str,
        persona_name: str,
        labels: list[str] | None = None,
    ) -> PersonaResponse:
        payload = safe_json_parse(strip_markdown_fences(raw))
        if not isinstance(payload, dict):
            logger.warning("Persona %s returned invalid JSON; using fallback.", persona_name)
            fallback_label = labels[0] if labels else "unknown"
            return PersonaResponse(
                persona_name=persona_name,
                label=fallback_label,
                confidence=0.0,
                reasoning=f"Failed to parse persona response as JSON: {raw[:200]}",
                key_factors=[],
            )

        return PersonaResponse(
            persona_name=persona_name,
            label=str(payload.get("label", "unknown")),
            confidence=clamp_confidence(payload.get("confidence", 0.0)),
            reasoning=str(payload.get("reasoning", "")),
            key_factors=[str(item) for item in payload.get("key_factors", [])],
            dissent_notes=(str(payload["dissent_notes"]) if "dissent_notes" in payload else None),
            raw_response=None,
            tokens_used=0,
            cost_usd=0.0,
        )

    def _consensus_reached(self, round_responses: list[PersonaResponse]) -> bool:
        labels = [response.label for response in round_responses]
        return bool(labels) and len(set(labels)) == 1
