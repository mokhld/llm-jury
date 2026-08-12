from __future__ import annotations

import unittest
from typing import Any

from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona


class _AlwaysFailingClient:
    """LLM client where every call raises (e.g. missing API key)."""

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("no api key")


def _personas() -> list[Persona]:
    return [
        Persona(name=name, role="role", system_prompt=name) for name in ("A", "B", "C")
    ]


class DegradedDebateTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end regression for the README quickstart with a broken LLM setup.

    Before the failed-vote fix, a missing API key produced a fabricated
    unanimous labels[0] verdict at confidence 1.0 — silently flipping an
    "unsafe" primary classification to "safe".
    """

    async def test_all_persona_failures_preserve_primary_result(self) -> None:
        classifier = FunctionClassifier(
            lambda text: ("unsafe", 0.55), ["safe", "unsafe"]
        )
        jury = Jury(
            classifier=classifier,
            personas=_personas(),
            confidence_threshold=0.7,
            judge=MajorityVoteJudge(),
            llm_client=_AlwaysFailingClient(),
        )

        verdict = await jury.classify("borderline message")

        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 0.55)
        self.assertTrue(verdict.was_escalated)
        self.assertEqual(verdict.persona_failures, 3)
        self.assertTrue(verdict.debate_degraded)

    async def test_healthy_verdict_is_not_degraded(self) -> None:
        classifier = FunctionClassifier(lambda text: ("safe", 0.95), ["safe", "unsafe"])
        jury = Jury(
            classifier=classifier,
            personas=_personas(),
            confidence_threshold=0.7,
            judge=MajorityVoteJudge(),
            llm_client=_AlwaysFailingClient(),
        )

        verdict = await jury.classify("clearly fine")

        self.assertFalse(verdict.was_escalated)
        self.assertEqual(verdict.persona_failures, 0)
        self.assertFalse(verdict.debate_degraded)

    async def test_to_dict_includes_degradation_fields(self) -> None:
        classifier = FunctionClassifier(
            lambda text: ("unsafe", 0.55), ["safe", "unsafe"]
        )
        jury = Jury(
            classifier=classifier,
            personas=_personas(),
            confidence_threshold=0.7,
            judge=MajorityVoteJudge(),
            llm_client=_AlwaysFailingClient(),
        )

        verdict = await jury.classify("borderline message")
        data = verdict.to_dict()

        self.assertEqual(data["persona_failures"], 3)
        self.assertTrue(data["debate_degraded"])


class BatchReturnExceptionsTests(unittest.IsolatedAsyncioTestCase):
    def _jury(self) -> Jury:
        def classify_fn(text: str):
            if text == "boom":
                raise RuntimeError("classifier exploded")
            return ("safe", 0.95)

        classifier = FunctionClassifier(classify_fn, ["safe", "unsafe"])
        return Jury(classifier=classifier, personas=[], confidence_threshold=0.7)

    async def test_default_raises_on_first_failure(self) -> None:
        jury = self._jury()
        with self.assertRaises(RuntimeError):
            await jury.classify_batch(["ok", "boom", "ok"], concurrency=1)

    async def test_return_exceptions_preserves_successful_verdicts(self) -> None:
        jury = self._jury()
        results = await jury.classify_batch(
            ["ok", "boom", "ok"], concurrency=1, return_exceptions=True
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].label, "safe")
        self.assertIsInstance(results[1], RuntimeError)
        self.assertEqual(results[2].label, "safe")


if __name__ == "__main__":
    unittest.main()
