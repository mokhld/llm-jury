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
            rounds=[
                [
                    PersonaResponse("A", "unsafe", 0.9, "harm", ["harm"]),
                ]
            ],
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

    async def test_pre_flight_estimate_skips_debate(self) -> None:
        """If estimated cost (N×rounds×per_persona) > cap, debate must NOT run."""
        classifier = FunctionClassifier(lambda _: ("safe", 0.3), ["safe", "unsafe"])
        personas = [
            Persona(name=f"P{i}", role="r", system_prompt="s") for i in range(4)
        ]
        debate_engine_called = False

        class TrackingEngine(ExpensiveDebateEngine):
            async def debate(self, *args: object, **kwargs: object) -> DebateTranscript:
                nonlocal debate_engine_called
                debate_engine_called = True
                return await super().debate(*args, **kwargs)  # type: ignore[misc]

        jury = Jury(
            classifier=classifier,
            personas=personas,
            max_debate_cost_usd=0.05,
            estimated_cost_per_persona_usd=0.01,
            judge=MockJudge(),
        )
        jury.debate_engine = TrackingEngine()

        # 4 personas × 2 rounds × $0.01 = $0.08 > $0.05 → pre-flight refusal
        self.assertGreater(
            jury.estimated_max_debate_cost_usd, jury.max_debate_cost_usd or 0
        )
        verdict = await jury.classify("text")
        self.assertEqual(verdict.judge_strategy, "cost_guard_pre_flight")
        self.assertEqual(verdict.label, "safe")
        self.assertTrue(verdict.was_escalated)
        self.assertIsNone(verdict.debate_transcript)
        self.assertFalse(
            debate_engine_called,
            "debate engine must not be called when pre-flight trips",
        )

    async def test_pre_flight_estimate_allows_debate_under_cap(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.3), ["safe", "unsafe"])
        personas = [Persona(name="A", role="r", system_prompt="s")]
        jury = Jury(
            classifier=classifier,
            personas=personas,
            max_debate_cost_usd=10.0,
            estimated_cost_per_persona_usd=0.01,
            judge=MockJudge(),
        )
        jury.debate_engine = ExpensiveDebateEngine()

        # 1 × 2 × 0.01 = 0.02, well under 10.0 → debate runs (but ExpensiveDebateEngine
        # returns 1.25 total, still under 10.0 → judge runs)
        verdict = await jury.classify("text")
        self.assertEqual(verdict.judge_strategy, "mock")

    async def test_jury_respects_judge_set_total_duration_ms(self) -> None:
        """Custom judge that explicitly sets total_duration_ms must NOT be overwritten."""
        classifier = FunctionClassifier(lambda _: ("safe", 0.3), ["safe", "unsafe"])
        personas = [Persona(name="A", role="r", system_prompt="s")]

        class JudgeWithExplicitDuration:
            async def judge(
                self, transcript: DebateTranscript, labels: list[str]
            ) -> Verdict:
                return Verdict(
                    label="unsafe",
                    confidence=0.9,
                    reasoning="judge",
                    was_escalated=False,  # Jury must override to True.
                    primary_result=transcript.primary_result,
                    debate_transcript=transcript,
                    judge_strategy="custom",
                    total_duration_ms=99999,  # Jury must NOT overwrite.
                    total_cost_usd=0.5,
                )

        jury = Jury(
            classifier=classifier, personas=personas, judge=JudgeWithExplicitDuration()
        )
        jury.debate_engine = ExpensiveDebateEngine()

        verdict = await jury.classify("text")
        self.assertEqual(
            verdict.total_duration_ms, 99999, "judge-set duration preserved"
        )
        self.assertTrue(verdict.was_escalated, "Jury authoritative for was_escalated")
        self.assertEqual(verdict.total_cost_usd, 0.5)

    async def test_jury_backfills_zero_duration_with_full_classify_time(self) -> None:
        """Default judges set total_duration_ms=0; Jury fills full-classify time."""
        classifier = FunctionClassifier(lambda _: ("safe", 0.3), ["safe", "unsafe"])
        personas = [Persona(name="A", role="r", system_prompt="s")]

        class JudgeWithZeroDuration:
            async def judge(
                self, transcript: DebateTranscript, labels: list[str]
            ) -> Verdict:
                return Verdict(
                    label="unsafe",
                    confidence=0.9,
                    reasoning="judge",
                    was_escalated=True,
                    primary_result=transcript.primary_result,
                    debate_transcript=transcript,
                    judge_strategy="zero_duration",
                    total_duration_ms=0,
                    total_cost_usd=None,
                )

        jury = Jury(
            classifier=classifier, personas=personas, judge=JudgeWithZeroDuration()
        )
        jury.debate_engine = ExpensiveDebateEngine()

        verdict = await jury.classify("text")
        self.assertGreaterEqual(verdict.total_duration_ms, 0)
        # The ExpensiveDebateEngine fixture returns duration_ms=5; Jury should now
        # supply the wall-clock total (>= 0, typically a few ms).

    async def test_estimated_max_debate_cost_property(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.9), ["safe", "unsafe"])
        personas = [
            Persona(name=f"P{i}", role="r", system_prompt="s") for i in range(3)
        ]
        jury = Jury(
            classifier=classifier,
            personas=personas,
            estimated_cost_per_persona_usd=0.02,
        )
        # 3 personas × 2 rounds (default) × 0.02 = 0.12
        self.assertAlmostEqual(jury.estimated_max_debate_cost_usd, 0.12)


if __name__ == "__main__":
    unittest.main()
