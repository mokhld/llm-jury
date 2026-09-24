from __future__ import annotations

import logging

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.debate.engine import DebateTranscript
from llm_jury.llm.client import LiteLLMClient, LLMClient
from llm_jury.personas.schema import build_judge_response_schema
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

from .base import JudgeStrategy, Verdict, _fallback_verdict, _usable_responses
from .majority_vote import _majority_vote

logger = logging.getLogger(__name__)


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


class LLMJudge(JudgeStrategy):
    DEFAULT_SYSTEM_PROMPT = (
        "You are the presiding judge in an expert panel. "
        "You have received assessments from multiple domain experts on a classification task.\n\n"
        "Your role is to:\n"
        "1. Weigh each expert's reasoning on its merits\n"
        "2. Consider the strength of evidence each expert cites\n"
        "3. Note where experts agree and disagree\n"
        "4. Factor in each expert's known perspective/bias\n"
        "5. If a debate summary is provided, use it to identify the decisive arguments\n"
        "6. Deliver a final classification with clear reasoning\n\n"
        "Respond ONLY with valid JSON:\n"
        "{\n"
        '  "label": "<final classification>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "reasoning": "<your synthesis of the debate>",\n'
        '  "key_agreements": ["<points all experts agreed on>"],\n'
        '  "key_disagreements": ["<points of contention>"],\n'
        '  "decisive_factor": "<what tipped the decision>"\n'
        "}"
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.llm_client = llm_client or LiteLLMClient()

    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        # If every persona call failed there is nothing to judge — skip the
        # LLM call (it would only see failure placeholders) and fall back to
        # the primary classifier result.
        if not any(_usable_responses(round_) for round_ in transcript.rounds):
            return _fallback_verdict(
                transcript,
                "llm_judge_fallback_personas_failed",
                "All persona calls failed; returning primary classifier result.",
            )

        prompt = self._build_prompt(transcript, labels)
        try:
            payload = await self.llm_client.complete(
                model=self.model,
                system_prompt=self.system_prompt,
                prompt=prompt,
                temperature=self.temperature,
                response_format=build_judge_response_schema(labels),
            )
        except Exception as exc:  # noqa: BLE001
            # A judge outage must not throw away the paid debate: decide by
            # vote instead.
            logger.warning(
                "LLM judge call failed; falling back to a majority vote: %s", exc
            )
            return self._vote_fallback(
                transcript,
                "llm_judge_fallback_error",
                f"LLM judge call failed ({type(exc).__name__}: {exc}).",
                judge_cost_usd=None,
            )

        judge_cost = payload_cost(payload)
        raw_content = str(payload.get("content") or "")
        data = safe_json_parse(strip_markdown_fences(raw_content))

        if data is None:
            return self._vote_fallback(
                transcript,
                "llm_judge_fallback_invalid_json",
                "LLM judge response was not valid JSON.",
                judge_cost,
            )

        raw_label = data.get("label")
        label = match_label(raw_label, labels)
        if label is None:
            return self._vote_fallback(
                transcript,
                "llm_judge_fallback_invalid_label",
                f"LLM judge returned label '{raw_label}', which is not one of the "
                "configured labels.",
                judge_cost,
            )

        raw_confidence = data.get("confidence")
        confidence = parse_confidence(raw_confidence)
        if confidence is None:
            return self._vote_fallback(
                transcript,
                "llm_judge_fallback_invalid_confidence",
                f"LLM judge returned confidence '{raw_confidence}', which is not a "
                "finite number.",
                judge_cost,
            )

        decisive_factor = data.get("decisive_factor")
        return Verdict(
            label=label,
            confidence=confidence,
            reasoning=str(data.get("reasoning", "LLM judge response.")),
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="llm_judge",
            total_duration_ms=0,  # Jury fills in the full-classify duration.
            total_cost_usd=add_costs(transcript.total_cost_usd, judge_cost),
            judge_details={
                "key_agreements": _as_str_list(data.get("key_agreements")),
                "key_disagreements": _as_str_list(data.get("key_disagreements")),
                "decisive_factor": (
                    str(decisive_factor) if decisive_factor is not None else None
                ),
            },
        )

    def _vote_fallback(
        self,
        transcript: DebateTranscript,
        strategy_name: str,
        reason: str,
        judge_cost_usd: float | None,
    ) -> Verdict:
        """Majority vote over the final round's valid responses, without an LLM.

        Used when the judge's own output is unusable. Falls back to the
        primary classifier result when the final round has no valid votes.
        """
        total_cost = add_costs(transcript.total_cost_usd, judge_cost_usd)
        final_round = (
            _usable_responses(transcript.rounds[-1]) if transcript.rounds else []
        )
        if not final_round:
            verdict = _fallback_verdict(
                transcript,
                strategy_name,
                f"{reason} No valid persona responses in the final round; "
                "returning primary classifier result.",
            )
            verdict.total_cost_usd = total_cost
            return verdict

        winner, confidence, vote_reasoning = _majority_vote(final_round)
        return Verdict(
            label=winner,
            confidence=confidence,
            reasoning=(
                f"{reason} Falling back to a majority vote over the final "
                f"round's persona responses. {vote_reasoning}"
            ),
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy=strategy_name,
            total_duration_ms=0,  # Jury fills in the full-classify duration.
            total_cost_usd=total_cost,
        )

    def _build_prompt(self, transcript, labels: list[str]) -> str:
        lines = [
            f"Input:\n{wrap_untrusted(transcript.input_text)}",
            f"Available labels: {', '.join(labels)}",
            f"Primary result: {transcript.primary_result.label} ({format_confidence(transcript.primary_result.confidence)})",
        ]

        persona_biases = getattr(transcript, "persona_biases", None) or {}
        if persona_biases:
            lines.append("\nExpert roster:")
            for name, bias in persona_biases.items():
                lines.append(f"- {name} (known bias: {bias})")

        for round_idx, round_responses in enumerate(transcript.rounds):
            valid = _usable_responses(round_responses)
            if not valid:
                continue
            heading = (
                "Initial Expert Opinions"
                if round_idx == 0
                else f"Revised Opinions (Round {round_idx + 1})"
            )
            lines.append(f"\n{heading}:")
            for response in valid:
                lines.append(
                    f"- {response.persona_name}: {response.label} ({response.confidence:.2f}) | "
                    f"Reasoning: {response.reasoning}"
                )

        if getattr(transcript, "summary", None):
            lines.append(f"\nDebate Summary:\n{transcript.summary}")

        lines.append(
            "\nRespond ONLY with JSON containing: "
            "label, confidence, reasoning, key_agreements, key_disagreements, decisive_factor."
        )
        return "\n".join(lines)
