from __future__ import annotations

from collections import Counter

from llm_jury.debate.engine import DebateTranscript

from .base import (
    _ALL_FAILED_REASON,
    JudgeStrategy,
    Verdict,
    _fallback_verdict,
    _usable_responses,
)


class MajorityVoteJudge(JudgeStrategy):
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        if not transcript.rounds or not transcript.rounds[-1]:
            return _fallback_verdict(transcript, "majority_vote")

        final_round = _usable_responses(transcript.rounds[-1])
        if not final_round:
            return _fallback_verdict(transcript, "majority_vote", _ALL_FAILED_REASON)

        counts = Counter(response.label for response in final_round)
        winner, winner_count = counts.most_common(1)[0]
        confidence = winner_count / len(final_round)
        reasons = [
            response.reasoning for response in final_round if response.label == winner
        ]

        return Verdict(
            label=winner,
            confidence=float(confidence),
            reasoning=(
                " ".join(reasons) if reasons else "Majority vote selected the winner."
            ),
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="majority_vote",
            total_duration_ms=0,  # Jury fills in the full-classify duration.
            total_cost_usd=transcript.total_cost_usd,
        )
