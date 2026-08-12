from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from llm_jury._version import __version__
from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.utils import json_serializable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # Number of persona calls across the debate that failed (LLM error or
    # unparseable output). Set authoritatively by Jury after judging.
    persona_failures: int = 0

    library_version: str = field(default_factory=lambda: __version__)
    created_at: str = field(default_factory=_utc_now_iso)

    @property
    def debate_degraded(self) -> bool:
        """True when at least one persona failed during the debate.

        Degraded verdicts were decided by fewer jurors than configured (or by
        none, in which case the primary classifier result was returned).
        Production pipelines can use this to route to human review.
        """
        return self.persona_failures > 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["debate_degraded"] = self.debate_degraded
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=json_serializable, ensure_ascii=True)


class JudgeStrategy(ABC):
    @abstractmethod
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        raise NotImplementedError


def _fallback_verdict(
    transcript: DebateTranscript,
    strategy_name: str,
    reason: str = "No persona responses; returning primary classifier result.",
) -> Verdict:
    """Build a fallback verdict from the primary result when no usable persona responses exist."""
    pr = transcript.primary_result
    return Verdict(
        label=pr.label,
        confidence=pr.confidence,
        reasoning=reason,
        was_escalated=True,
        primary_result=pr,
        debate_transcript=transcript,
        judge_strategy=strategy_name,
        total_duration_ms=0,  # Jury fills in the full-classify duration.
        total_cost_usd=getattr(transcript, "total_cost_usd", None),
    )


def _usable_responses(responses: list) -> list:
    """Responses that carry a real vote — failed persona placeholders are excluded."""
    return [r for r in responses if not getattr(r, "failed", False)]


_ALL_FAILED_REASON = (
    "All persona calls in the final round failed; "
    "returning primary classifier result."
)
