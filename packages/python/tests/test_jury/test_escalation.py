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
                PersonaResponse("B", "unsafe", 0.8, "risk", ["risk"]),
            ]],
            duration_ms=20,
            total_tokens=100,
            total_cost_usd=0.02,
        )


class MockJudge:
    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        return Verdict(
            label="unsafe",
            confidence=0.9,
            reasoning="majority unsafe",
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="mock",
            total_duration_ms=transcript.duration_ms,
            total_cost_usd=transcript.total_cost_usd,
        )


class JuryEscalationTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_confidence_triggers_escalation(self) -> None:
        classifier = FunctionClassifier(lambda text: ("safe", 0.4), ["safe", "unsafe"])
        personas = [Persona(name="A", role="role", system_prompt="prompt")]

        escalations: list[str] = []
        verdicts: list[str] = []

        jury = Jury(
            classifier=classifier,
            personas=personas,
            confidence_threshold=0.7,
            judge=MockJudge(),
            on_escalation=lambda text, result: escalations.append(text),
            on_verdict=lambda verdict: verdicts.append(verdict.label),
        )
        jury.debate_engine = MockDebateEngine()

        verdict = await jury.classify("ambiguous")

        self.assertTrue(verdict.was_escalated)
        self.assertEqual(verdict.label, "unsafe")
        self.assertEqual(jury.stats.escalated, 1)
        self.assertEqual(escalations, ["ambiguous"])
        self.assertEqual(verdicts, ["unsafe"])


if __name__ == "__main__":
    unittest.main()
