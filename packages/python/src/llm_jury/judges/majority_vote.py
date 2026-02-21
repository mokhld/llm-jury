from __future__ import annotations

from collections import Counter

from llm_jury.debate.engine import DebateTranscript

from .base import JudgeStrategy, Verdict, _fallback_verdict


class MajorityVoteJudge(JudgeStrategy):
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        if not transcript.rounds or not transcript.rounds[-1]:
            return _fallback_verdict(transcript, "majority_vote")

        final_round = transcript.rounds[-1]
        counts = Counter(response.label for response in final_round)
        winner, winner_count = counts.most_common(1)[0]
        confidence = winner_count / len(final_round)
        reasons = [response.reasoning for response in final_round if response.label == winner]

        return Verdict(
            label=winner,
            confidence=float(confidence),
            reasoning=" ".join(reasons) if reasons else "Majority vote selected the winner.",
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="majority_vote",
            total_duration_ms=transcript.duration_ms,
            total_cost_usd=transcript.total_cost_usd,
        )
