from __future__ import annotations

from collections import Counter

from llm_jury.debate.engine import DebateTranscript
from llm_jury.personas.base import PersonaResponse

from .base import (
    _ALL_FAILED_REASON,
    JudgeStrategy,
    Verdict,
    _fallback_verdict,
    _usable_responses,
)


def _majority_vote(responses: list[PersonaResponse]) -> tuple[str, float, str]:
    """Return ``(label, confidence, reasoning)`` for a non-empty list of votes.

    Confidence is the winner's share of the votes. Ties go to the label seen
    first.
    """
    counts = Counter(response.label for response in responses)
    winner, winner_count = counts.most_common(1)[0]
    confidence = winner_count / len(responses)
    reasons = [response.reasoning for response in responses if response.label == winner]
    reasoning = " ".join(reasons) if reasons else "Majority vote selected the winner."
    return winner, float(confidence), reasoning


class MajorityVoteJudge(JudgeStrategy):
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        if not transcript.rounds or not transcript.rounds[-1]:
            return _fallback_verdict(transcript, "majority_vote")

        final_round = _usable_responses(transcript.rounds[-1])
        if not final_round:
            return _fallback_verdict(transcript, "majority_vote", _ALL_FAILED_REASON)

        winner, confidence, reasoning = _majority_vote(final_round)

        return Verdict(
            label=winner,
            confidence=confidence,
            reasoning=reasoning,
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="majority_vote",
            total_duration_ms=0,  # Jury fills in the full-classify duration.
            total_cost_usd=transcript.total_cost_usd,
        )
