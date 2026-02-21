from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.utils import json_serializable


@dataclass(slots=True)
class Verdict:
    label: str
    confidence: float
    reasoning: str

    was_escalated: bool
    primary_result: ClassificationResult
    debate_transcript: DebateTranscript | None
    judge_strategy: str

    total_duration_ms: int
    total_cost_usd: float | None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=json_serializable, ensure_ascii=True)


class JudgeStrategy(ABC):
    @abstractmethod
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        raise NotImplementedError


def _fallback_verdict(transcript: DebateTranscript, strategy_name: str) -> Verdict:
    """Build a fallback verdict from the primary result when no persona responses exist."""
    pr = transcript.primary_result
    return Verdict(
        label=pr.label,
        confidence=pr.confidence,
        reasoning="No persona responses; returning primary classifier result.",
        was_escalated=True,
        primary_result=pr,
        debate_transcript=transcript,
        judge_strategy=strategy_name,
        total_duration_ms=getattr(transcript, "duration_ms", 0),
        total_cost_usd=getattr(transcript, "total_cost_usd", None),
    )
