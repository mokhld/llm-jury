from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.classifiers.base import ClassificationResult
from llm_jury.llm.client import LiteLLMClient, LLMClient
from llm_jury.personas.base import Persona, PersonaResponse
from llm_jury.personas.schema import build_persona_response_schema
from llm_jury.utils import (
    add_costs,
    format_confidence,
    match_label,
    parse_confidence,
    payload_cost,
    safe_json_parse,
    strip_markdown_fences,
    wrap_untrusted,
)

logger = logging.getLogger(__name__)

# Absorbs float noise so spend that lands exactly on the cap does not trip it.
COST_CAP_TOLERANCE_USD = 1e-9


def guard_spend(
    known_cost_usd: float | None,
    unpriced_calls: int,
    estimated_cost_per_call_usd: float | None,
) -> float:
    """Spend the cost guard compares against ``max_debate_cost_usd``.

    Known cost so far plus the per-call estimate for every call that reported
    no cost, so a client that never reports cost cannot slip past the cap.
    """
    return (known_cost_usd or 0.0) + unpriced_calls * (
        estimated_cost_per_call_usd or 0.0
    )


def exceeds_cost_cap(spend_usd: float, max_cost_usd: float | None) -> bool:
    """True when ``spend_usd`` is over the cap (``None`` means no cap)."""
    if max_cost_usd is None:
        return False
    return spend_usd > max_cost_usd + COST_CAP_TOLERANCE_USD


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
    # F7: optional high-confidence early stop for the DELIBERATION
    # debate loop. When set, the loop also exits when the MIN
    # persona confidence in a round is >= this threshold, even if
    # personas disagree on label. None disables (default behaviour:
    # only unanimous-label consensus halts early).
    early_stop_min_confidence: float | None = None


@dataclass(slots=True)
class DebateTranscript:
    input_text: str
    primary_result: ClassificationResult
    rounds: list[list[PersonaResponse]]
    duration_ms: int
    total_tokens: int
    # Sum of every persona and summariser call that reported a cost. None only
    # when no call reported one.
    total_cost_usd: float | None
    summary: str | None = None
    # LLM calls in the debate (persona and summariser, including calls that
    # raised) that reported no cost. total_cost_usd leaves them out, so a
    # non-zero count means the total is a lower bound.
    unpriced_calls: int = 0
    # Persona name -> known_bias, for the personas that declare one.
    persona_biases: dict[str, str] = field(default_factory=dict)

    @property
    def persona_failures(self) -> int:
        """Number of persona calls across all rounds that failed."""
        return sum(1 for round_ in self.rounds for r in round_ if r.failed)


def _valid_responses(responses: list[PersonaResponse]) -> list[PersonaResponse]:
    """Responses that carry a real vote (persona call and parse succeeded)."""
    return [r for r in responses if not r.failed]


@dataclass(slots=True)
class _DebateSpend:
    """Running token and cost totals for one debate."""

    tokens: int = 0
    cost_usd: float | None = None
    unpriced_calls: int = 0

    def record(self, tokens: int, cost_usd: float | None) -> None:
        self.tokens += tokens
        if cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd = add_costs(self.cost_usd, cost_usd)

    def record_responses(self, responses: list[PersonaResponse]) -> None:
        for response in responses:
            self.record(response.tokens_used, response.cost_usd)


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
        estimated_cost_per_call_usd: float | None = None,
    ) -> DebateTranscript:
        """Run the debate and return its transcript.

        ``max_cost_usd`` caps spend mid-flight: once the guard spend (known
        cost plus ``estimated_cost_per_call_usd`` for every call that reported
        no cost) goes over the cap, no further persona round or summariser
        call starts.
        """
        start = time.perf_counter()
        rounds: list[list[PersonaResponse]] = []
        spend = _DebateSpend()
        summary: str | None = None
        persona_biases = {
            persona.name: persona.known_bias
            for persona in self.personas
            if persona.known_bias
        }

        def over_cap() -> bool:
            return exceeds_cost_cap(
                guard_spend(
                    spend.cost_usd, spend.unpriced_calls, estimated_cost_per_call_usd
                ),
                max_cost_usd,
            )

        def build_transcript() -> DebateTranscript:
            return DebateTranscript(
                input_text=text,
                primary_result=primary_result,
                rounds=rounds,
                duration_ms=int((time.perf_counter() - start) * 1000),
                total_tokens=spend.tokens,
                total_cost_usd=spend.cost_usd,
                summary=summary,
                unpriced_calls=spend.unpriced_calls,
                persona_biases=persona_biases,
            )

        if not self.personas:
            # No calls were made, so the cost is known to be zero.
            return DebateTranscript(
                input_text=text,
                primary_result=primary_result,
                rounds=[],
                duration_ms=int((time.perf_counter() - start) * 1000),
                total_tokens=0,
                total_cost_usd=0.0,
            )

        if self.config.mode in (DebateMode.INDEPENDENT, DebateMode.ADVERSARIAL):
            responses = await self._run_round(
                text, primary_result, labels, prior_rounds=[]
            )
            rounds.append(responses)
            spend.record_responses(responses)

        elif self.config.mode == DebateMode.SEQUENTIAL:
            responses: list[PersonaResponse] = []
            for persona in self.personas:
                try:
                    response = await self._query_persona(
                        persona=persona,
                        text=text,
                        primary_result=primary_result,
                        labels=labels,
                        prior_rounds=[responses] if responses else [],
                    )
                except (
                    Exception
                ) as exc:  # noqa: BLE001 — degrade gracefully on any persona failure
                    logger.warning(
                        "Persona %s failed during sequential debate: %s",
                        persona.name,
                        exc,
                    )
                    response = self._failed_persona_response(persona, exc, labels)
                responses.append(response)
                spend.record(response.tokens_used, response.cost_usd)
                if over_cap():
                    break
            rounds.append(responses)

        elif self.config.mode == DebateMode.DELIBERATION:
            # Stage 1: Initial opinions (parallel, independent)
            first_round = await self._run_round(
                text, primary_result, labels, prior_rounds=[]
            )
            rounds.append(first_round)
            spend.record_responses(first_round)

            if over_cap():
                return build_transcript()

            # If every persona call failed (bad API key, provider outage),
            # further rounds and the summariser are doomed too — stop paying
            # for them. The failed round stays in the transcript for audit.
            if not _valid_responses(first_round):
                logger.warning(
                    "All %d persona calls failed in the opening round; "
                    "aborting debate early.",
                    len(first_round),
                )
                return build_transcript()

            # Consensus in the opening round (unanimous labels, or the
            # early_stop_min_confidence rule) settles the debate: skip the
            # deliberation rounds and the summariser.
            if self._consensus_reached(first_round):
                return build_transcript()

            # Stage 2: Structured debate rounds (personas engage with prior opinions)
            for _ in range(1, max(1, self.config.max_rounds)):
                current = await self._run_deliberation_round(
                    text,
                    primary_result,
                    labels,
                    prior_rounds=rounds,
                )
                rounds.append(current)
                spend.record_responses(current)

                if over_cap():
                    break
                if not _valid_responses(current):
                    logger.warning(
                        "All %d persona calls failed in a deliberation round; "
                        "halting further rounds.",
                        len(current),
                    )
                    break
                if self._consensus_reached(current):
                    break

            # Stage 3: Summarisation — degrade gracefully if the summariser
            # call fails. The persona rounds are the load-bearing output; a
            # missing synthesis must not crash the verdict.
            if not over_cap():
                try:
                    summary, s_tokens, s_cost = await self._summarise(
                        text, labels, rounds
                    )
                    spend.record(s_tokens, s_cost)
                except (
                    Exception
                ) as exc:  # noqa: BLE001 — same rationale as per-persona fallback
                    logger.warning(
                        "Summarisation failed; returning transcript without summary: %s",
                        exc,
                    )
                    # The failed call may still have been billed; its cost is unknown.
                    spend.record(0, None)
                    summary = None

        return build_transcript()

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
                return await self._query_persona(
                    persona, text, primary_result, labels, prior_rounds
                )

        results = await asyncio.gather(
            *[_wrapped(persona) for persona in self.personas],
            return_exceptions=True,
        )
        return self._gather_with_fallback(results, labels)

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
                    persona,
                    text,
                    primary_result,
                    labels,
                    prior_rounds,
                )

        results = await asyncio.gather(
            *[_wrapped(persona) for persona in self.personas],
            return_exceptions=True,
        )
        return self._gather_with_fallback(results, labels)

    def _gather_with_fallback(
        self,
        results: list[PersonaResponse | BaseException],
        labels: list[str],
    ) -> list[PersonaResponse]:
        out: list[PersonaResponse] = []
        for persona, result in zip(self.personas, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "Persona %s failed during debate round: %s",
                    persona.name,
                    result,
                )
                out.append(self._failed_persona_response(persona, result, labels))
            else:
                out.append(result)
        return out

    @staticmethod
    def _failed_persona_response(
        persona: Persona,
        error: BaseException,
        labels: list[str],
    ) -> PersonaResponse:
        fallback_label = labels[0] if labels else "unknown"
        return PersonaResponse(
            persona_name=persona.name,
            label=fallback_label,
            confidence=0.0,
            reasoning=f"Persona call failed: {type(error).__name__}: {error}",
            key_factors=[],
            failed=True,
        )

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
        prompt = self._build_persona_prompt(
            persona, text, primary_result, labels, prior_rounds
        )
        return await self._call_persona(persona, prompt, labels)

    async def _query_persona_deliberation(
        self,
        persona: Persona,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        prior_rounds: list[list[PersonaResponse]],
    ) -> PersonaResponse:
        prompt = self._build_deliberation_prompt(
            persona, text, primary_result, labels, prior_rounds
        )
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
            response_format=build_persona_response_schema(labels),
        )
        raw_content = str(payload.get("content") or "")
        response = self._parse_persona_response(raw_content, persona.name, labels)
        response.raw_response = raw_content
        response.tokens_used = int(payload.get("tokens", 0) or 0)
        response.cost_usd = payload_cost(payload)
        return response

    # ------------------------------------------------------------------
    # Summarisation (Stage 3)
    # ------------------------------------------------------------------

    async def _summarise(
        self,
        text: str,
        labels: list[str],
        rounds: list[list[PersonaResponse]],
    ) -> tuple[str, int, float | None]:
        """Produce a structured summary of the debate.

        Returns ``(summary, tokens, cost)``; cost is ``None`` when the client
        reported none.
        """
        parts = [
            f"## Input\n\n{wrap_untrusted(text)}\n",
            f"## Labels\n\n{', '.join(labels)}\n",
        ]

        for r_idx, round_responses in enumerate(rounds):
            valid = _valid_responses(round_responses)
            if not valid:
                continue
            heading = (
                "Initial Expert Opinions"
                if r_idx == 0
                else f"Revised Opinions (Round {r_idx + 1})"
            )
            parts.append(f"## {heading}\n")
            for resp in valid:
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
        return summary_text, tokens, payload_cost(payload)

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
        parts = [
            f"## Persona\n\n{persona.name}: {persona.role}\n",
            f"## Input to Classify\n\n{wrap_untrusted(text)}\n",
        ]
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
                f" (confidence: {format_confidence(primary.confidence)})"
                if self.config.include_confidence
                else ""
            )
            parts.append(
                "## Primary Classifier Result\n\n"
                f"Label: {primary.label}{confidence_suffix}\n"
                "Note: This was flagged as low-confidence and escalated to you.\n"
            )

        if prior_rounds:
            parts.append("## Previous Assessments\n")
            for idx, round_responses in enumerate(prior_rounds):
                valid = _valid_responses(round_responses)
                if not valid:
                    continue
                parts.append(f"\n### Round {idx + 1}\n")
                for response in valid:
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
        parts = [
            f"## Persona\n\n{persona.name}: {persona.role}\n",
            f"## Input to Classify\n\n{wrap_untrusted(text)}\n",
        ]
        parts.append(f"## Available Labels\n\n{', '.join(labels)}\n")

        if self.config.include_primary_result:
            confidence_suffix = (
                f" (confidence: {format_confidence(primary.confidence)})"
                if self.config.include_confidence
                else ""
            )
            parts.append(
                "## Primary Classifier Result\n\n"
                f"Label: {primary.label}{confidence_suffix}\n"
                "Note: This was flagged as low-confidence and escalated to you.\n"
            )

        if prior_rounds:
            first_valid = _valid_responses(prior_rounds[0])
            if first_valid:
                parts.append("## Initial Expert Opinions\n")
                for response in first_valid:
                    parts.append(
                        f"**{response.persona_name}**: {response.label} (confidence: {response.confidence:.2f})\n"
                        f"Reasoning: {response.reasoning}\n"
                    )

            for r_idx in range(1, len(prior_rounds)):
                valid = _valid_responses(prior_rounds[r_idx])
                if not valid:
                    continue
                parts.append(f"\n## Revised Opinions (Round {r_idx + 1})\n")
                for response in valid:
                    parts.append(
                        f"**{response.persona_name}**: {response.label} (confidence: {response.confidence:.2f})\n"
                        f"Reasoning: {response.reasoning}\n"
                    )

        parts.append(
            f"\n## Deliberation Instructions\n\n{_DELIBERATION_INSTRUCTIONS}\n"
        )
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
        """Parse a persona reply into a vote.

        Output that is not JSON, names a label outside ``labels`` or carries
        a confidence that is not a finite number becomes a ``failed``
        placeholder: it stays in the transcript for audit but casts no vote.
        Matched labels are returned in their configured spelling.
        """
        fallback_label = labels[0] if labels else "unknown"

        def failed(problem: str, reason: str) -> PersonaResponse:
            logger.warning(
                "Persona %s returned %s; recording a failed response.",
                persona_name,
                problem,
            )
            return PersonaResponse(
                persona_name=persona_name,
                label=fallback_label,
                confidence=0.0,
                reasoning=reason,
                key_factors=[],
                raw_response=raw,
                failed=True,
            )

        payload = safe_json_parse(strip_markdown_fences(raw))
        if not isinstance(payload, dict):
            return failed(
                "invalid JSON",
                f"Failed to parse persona response as JSON: {raw[:200]}",
            )

        raw_label = payload.get("label")
        label = match_label(raw_label, labels or [])
        if label is None:
            return failed(
                "a label outside the configured labels",
                f"Persona returned label '{raw_label}', which is not one of the "
                "configured labels.",
            )

        raw_confidence = payload.get("confidence")
        confidence = parse_confidence(raw_confidence)
        if confidence is None:
            return failed(
                "an invalid confidence",
                f"Persona returned confidence '{raw_confidence}', which is not a "
                "finite number.",
            )

        dissent_raw = payload.get("dissent_notes")
        key_factors_raw = payload.get("key_factors")
        return PersonaResponse(
            persona_name=persona_name,
            label=label,
            confidence=confidence,
            reasoning=str(payload.get("reasoning", "")),
            key_factors=(
                [str(item) for item in key_factors_raw]
                if isinstance(key_factors_raw, list)
                else []
            ),
            dissent_notes=str(dissent_raw) if dissent_raw is not None else None,
            raw_response=None,
            tokens_used=0,
            cost_usd=None,
        )

    def _consensus_reached(self, round_responses: list[PersonaResponse]) -> bool:
        # Failed responses are placeholders, not votes: a round where two
        # personas agree and a third errored is real consensus, and a round
        # of pure failures is not unanimous agreement on labels[0].
        valid = _valid_responses(round_responses)
        if not valid:
            return False
        labels = [response.label for response in valid]
        if len(set(labels)) == 1:
            return True
        # F7: high-confidence early stop. When every persona this
        # round is highly confident in its own answer, further
        # deliberation rarely changes the verdict — let the judge
        # break the tie now instead of paying for another round.
        threshold = self.config.early_stop_min_confidence
        if threshold is not None:
            min_confidence = min(r.confidence for r in valid)
            if min_confidence >= threshold:
                return True
        return False
