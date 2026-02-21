from __future__ import annotations

from collections import defaultdict

from llm_jury.debate.engine import DebateTranscript

from .base import JudgeStrategy, Verdict, _fallback_verdict


class WeightedVoteJudge(JudgeStrategy):
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        if not transcript.rounds or not transcript.rounds[-1]:
            return _fallback_verdict(transcript, "weighted_vote")

        final_round = transcript.rounds[-1]
        scores: dict[str, float] = defaultdict(float)

        for response in final_round:
            scores[response.label] += float(response.confidence)

        winner = max(scores, key=lambda k: scores[k])
        total = sum(scores.values()) or 1.0
        confidence = float(scores[winner] / total)

        return Verdict(
            label=winner,
            confidence=confidence,
            reasoning="Weighted vote based on persona confidence scores.",
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="weighted_vote",
            total_duration_ms=transcript.duration_ms,
            total_cost_usd=transcript.total_cost_usd,
        )
