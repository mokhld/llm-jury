from __future__ import annotations

import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.base import Verdict
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona, PersonaResponse


class ExpensiveDebateEngine:
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
            total_tokens=200,
            total_cost_usd=1.25,
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
            total_cost_usd=0.1,
        )


class JuryControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_debate_concurrency_is_configurable(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.9), ["safe", "unsafe"])
        jury = Jury(
            classifier=classifier,
            personas=[Persona(name="A", role="r", system_prompt="s")],
            debate_concurrency=2,
            judge=MockJudge(),
        )
        self.assertEqual(jury.debate_engine.concurrency, 2)

    async def test_escalation_override_false_skips_debate(self) -> None:
        classifier = FunctionClassifier(lambda _: ("unsafe", 0.2), ["safe", "unsafe"])
        jury = Jury(
            classifier=classifier,
            personas=[Persona(name="A", role="r", system_prompt="s")],
            escalation_override=lambda result: False,
            judge=MockJudge(),
        )

        verdict = await jury.classify("text")
        self.assertFalse(verdict.was_escalated)
        self.assertEqual(jury.stats.fast_path, 1)

    async def test_escalation_override_true_forces_debate(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.95), ["safe", "unsafe"])
        jury = Jury(
            classifier=classifier,
            personas=[Persona(name="A", role="r", system_prompt="s")],
            escalation_override=lambda result: True,
            judge=MockJudge(),
        )
        jury.debate_engine = ExpensiveDebateEngine()

        verdict = await jury.classify("text")
        self.assertTrue(verdict.was_escalated)
        self.assertEqual(jury.stats.escalated, 1)

    async def test_max_cost_fallback(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.3), ["safe", "unsafe"])
        jury = Jury(
            classifier=classifier,
            personas=[Persona(name="A", role="r", system_prompt="s")],
            max_debate_cost_usd=0.5,
            judge=MockJudge(),
        )
        jury.debate_engine = ExpensiveDebateEngine()

        verdict = await jury.classify("text")
        self.assertEqual(verdict.judge_strategy, "cost_guard_primary_fallback")
        self.assertEqual(verdict.label, "safe")


if __name__ == "__main__":
    unittest.main()
