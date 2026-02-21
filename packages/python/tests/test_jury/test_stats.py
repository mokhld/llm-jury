from __future__ import annotations

import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.base import Verdict
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona, PersonaResponse


class MockDebateEngine:
    async def debate(
        self,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        max_cost_usd: float | None = None,
    ) -> DebateTranscript:
        return DebateTranscript(
            input_text=text,
            primary_result=primary_result,
            rounds=[[
                PersonaResponse("A", "unsafe", 0.9, "harm", ["harm"]),
            ]],
            duration_ms=5,
            total_tokens=10,
            total_cost_usd=0.001,
        )


class MockJudge:
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        return Verdict(
            label="unsafe",
            confidence=0.9,
            reasoning="mock",
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="mock",
            total_duration_ms=5,
            total_cost_usd=0.001,
        )


class JuryStatsTests(unittest.IsolatedAsyncioTestCase):
    async def test_stats_tracking(self) -> None:
        calls = [("safe", 0.9), ("unsafe", 0.3), ("safe", 0.95), ("unsafe", 0.2)]

        def classify_fn(text: str):
            return calls.pop(0)

        classifier = FunctionClassifier(classify_fn, ["safe", "unsafe"])
        jury = Jury(
            classifier=classifier,
            personas=[Persona(name="A", role="role", system_prompt="prompt")],
            confidence_threshold=0.7,
            judge=MockJudge(),
        )
        jury.debate_engine = MockDebateEngine()

        for text in ["a", "b", "c", "d"]:
            await jury.classify(text)

        self.assertEqual(jury.stats.total, 4)
        self.assertEqual(jury.stats.fast_path, 2)
        self.assertEqual(jury.stats.escalated, 2)
        self.assertAlmostEqual(jury.stats.escalation_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
