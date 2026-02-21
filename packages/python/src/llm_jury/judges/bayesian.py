from __future__ import annotations

from collections import defaultdict

from llm_jury.debate.engine import DebateTranscript

from .base import JudgeStrategy, Verdict, _fallback_verdict


class BayesianJudge(JudgeStrategy):
    def __init__(self, persona_priors: dict[str, dict[str, float]] | None = None) -> None:
        self.persona_priors = persona_priors or {}

    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        if not transcript.rounds or not transcript.rounds[-1]:
            return _fallback_verdict(transcript, "bayesian")

        posterior: dict[str, float] = defaultdict(lambda: 1.0)
        for response in transcript.rounds[-1]:
            for label in labels:
                prior = self.persona_priors.get(response.persona_name, {}).get(label, 1.0 / max(1, len(labels)))
                likelihood = response.confidence if response.label == label else (1.0 - response.confidence)
                posterior[label] *= max(1e-6, prior * likelihood)

        total = sum(posterior.values()) or 1.0
        for label in list(posterior.keys()):
            posterior[label] /= total

        winner = max(posterior, key=lambda k: posterior[k])
        return Verdict(
            label=winner,
            confidence=float(posterior[winner]),
            reasoning="Bayesian aggregation across persona responses.",
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="bayesian",
            total_duration_ms=transcript.duration_ms,
            total_cost_usd=transcript.total_cost_usd,
        )
