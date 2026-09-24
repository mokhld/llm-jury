"""Shared evaluation fixture.

packages/typescript/tests/evaluation/fixture.ts holds the same rows, and both
SDKs' tests assert the same hand-computed numbers against them.
"""

from __future__ import annotations

from llm_jury.classifiers.base import ClassificationResult, Classifier
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.base import Verdict

LABELS = ["safe", "unsafe"]

# (text, expected, primary label, primary confidence,
#  jury label, jury cost, unpriced calls, persona failures, judge strategy,
#  duration ms). Items at or above band_upper=0.95 have no jury script.
JUDGED = "llm_judge"
ERROR = "llm_judge_fallback_error"
ALL_FAILED = "llm_judge_fallback_personas_failed"
FIXTURE: list[tuple] = [
    ("t1", "safe", "safe", 0.98, None, None, 0, 0, None, None),
    ("t2", "unsafe", "safe", 0.96, None, None, 0, 0, None, None),
    ("t3", "unsafe", "safe", 0.92, "unsafe", 0.25, 0, 0, JUDGED, 100),
    ("t4", "safe", "safe", 0.85, "safe", 0.25, 0, 0, JUDGED, 200),
    ("t5", "safe", "unsafe", 0.75, "safe", 0.25, 0, 1, JUDGED, 300),
    ("t6", "unsafe", "unsafe", 0.70, "safe", 0.25, 0, 0, ERROR, 400),
    ("t7", "safe", "unsafe", 0.60, "unsafe", None, 3, 3, ALL_FAILED, 500),
    ("t8", "unsafe", "unsafe", 0.55, "unsafe", 0.25, 0, 0, JUDGED, 600),
    ("t9", "safe", "unsafe", 0.45, "safe", 0.25, 0, 0, JUDGED, 700),
    ("t10", "unsafe", "safe", 0.40, "unsafe", 0.25, 0, 0, JUDGED, 800),
]
TEXTS = [row[0] for row in FIXTURE]
EXPECTED = [row[1] for row in FIXTURE]
SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


class CountingClassifier(Classifier):
    """Replays stored predictions and counts calls per text."""

    def __init__(self, predictions: dict[str, tuple[str, float]]) -> None:
        self.labels = LABELS
        self.predictions = predictions
        self.calls: dict[str, int] = {}

    async def classify(self, text: str) -> ClassificationResult:
        self.calls[text] = self.calls.get(text, 0) + 1
        label, confidence = self.predictions[text]
        return ClassificationResult(label, confidence, cost_usd=0.0)


def _other(label: str) -> str:
    return "unsafe" if label == "safe" else "safe"


class ScriptedJury:
    """A jury whose escalations return scripted verdicts.

    ``mode`` "script" uses the fixture's jury columns; "right" and "wrong"
    answer the expected label or the other one for every item.
    """

    def __init__(self, mode: str = "script", rows: list[tuple] | None = None) -> None:
        rows = rows if rows is not None else FIXTURE
        self.classifier = CountingClassifier({r[0]: (r[2], r[3]) for r in rows})
        self.personas = ["p"]
        self.threshold = 0.7
        self.rows = {r[0]: r for r in rows}
        self.mode = mode
        self.escalated: list[str] = []

    async def escalate(self, text: str, primary: ClassificationResult) -> Verdict:
        self.escalated.append(text)
        _, expected, _, _, label, cost, unpriced, failures, strategy, duration = (
            self.rows[text]
        )
        if self.mode == "right":
            label, cost, unpriced, failures = expected, 0.25, 0, 0
        elif self.mode == "wrong":
            label, cost, unpriced, failures = _other(expected), 0.25, 0, 0
        transcript = DebateTranscript(
            input_text=text,
            primary_result=primary,
            rounds=[],
            duration_ms=duration or 0,
            total_tokens=0,
            total_cost_usd=cost,
            unpriced_calls=unpriced,
        )
        total = None if cost is None else primary.cost_usd + cost
        return Verdict(
            label=label,
            confidence=0.8,
            reasoning="scripted",
            was_escalated=True,
            primary_result=primary,
            debate_transcript=transcript,
            judge_strategy=strategy or JUDGED,
            total_duration_ms=duration or 0,
            total_cost_usd=total,
            persona_failures=failures,
        )
