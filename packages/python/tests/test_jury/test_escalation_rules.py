from __future__ import annotations

import json
import unittest
from typing import Any

from llm_jury.classifiers.base import ClassificationResult, Classifier
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.classifiers.llm_classifier import LLMClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona
from tests.helpers import FakeLLMClient, FakeLLMReply

LABELS = ["safe", "unsafe"]


def _persona_reply(label: str = "unsafe", confidence: float = 0.9) -> FakeLLMReply:
    return FakeLLMReply(
        json.dumps(
            {
                "label": label,
                "confidence": confidence,
                "reasoning": "persona reasoning",
                "key_factors": ["k"],
            }
        )
    )


def _personas() -> list[Persona]:
    return [
        Persona(name=n, role="role", system_prompt=f"persona-{n}", model="persona")
        for n in ("A", "B", "C")
    ]


class _StaticClassifier(Classifier):
    def __init__(self, result: ClassificationResult) -> None:
        self.labels = LABELS
        self.result = result

    async def classify(self, text: str) -> ClassificationResult:
        return self.result


class NonFiniteConfidenceEscalationTests(unittest.IsolatedAsyncioTestCase):
    """BUG-02: a primary confidence that is not a finite number escalates."""

    def _jury(self, classifier: Classifier, **kwargs: Any) -> Jury:
        return Jury(
            classifier=classifier,
            personas=_personas(),
            confidence_threshold=0.7,
            judge=MajorityVoteJudge(),
            llm_client=FakeLLMClient({"persona": _persona_reply()}),
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
            **kwargs,
        )

    async def test_nan_confidence_escalates(self) -> None:
        jury = self._jury(FunctionClassifier(lambda _: ("safe", float("nan")), LABELS))
        verdict = await jury.classify("text")
        self.assertTrue(verdict.was_escalated)
        self.assertEqual(verdict.label, "unsafe")

    async def test_infinite_confidence_escalates(self) -> None:
        jury = self._jury(FunctionClassifier(lambda _: ("safe", float("inf")), LABELS))
        verdict = await jury.classify("text")
        self.assertTrue(verdict.was_escalated)

    async def test_none_confidence_escalates(self) -> None:
        jury = self._jury(_StaticClassifier(ClassificationResult("safe", None)))  # type: ignore[arg-type]
        verdict = await jury.classify("text")
        self.assertTrue(verdict.was_escalated)

    async def test_escalation_override_still_wins(self) -> None:
        jury = self._jury(
            FunctionClassifier(lambda _: ("safe", float("nan")), LABELS),
            escalation_override=lambda _result: False,
        )
        verdict = await jury.classify("text")
        self.assertFalse(verdict.was_escalated)

    async def test_llm_classifier_nan_confidence_escalates(self) -> None:
        # json.loads accepts NaN; it used to clamp to 1.0 and skip escalation.
        llm = FakeLLMClient(
            {
                "primary": FakeLLMReply('{"label": "safe", "confidence": NaN}'),
                "persona": _persona_reply(),
            }
        )
        classifier = LLMClassifier(model="primary", labels=LABELS, llm_client=llm)
        jury = Jury(
            classifier=classifier,
            personas=_personas(),
            judge=MajorityVoteJudge(),
            llm_client=llm,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        verdict = await jury.classify("text")

        self.assertTrue(verdict.was_escalated)
        self.assertEqual(
            verdict.primary_result.raw_output["error"], "invalid_confidence"
        )


class ThresholdValidationTests(unittest.TestCase):
    """BUG-02: the Jury rejects a threshold that is not a finite number in [0, 1]."""

    def test_invalid_thresholds_raise(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.9), LABELS)
        for bad in (float("nan"), float("inf"), -0.1, 1.5, "0.7", None, True):
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                Jury(classifier=classifier, personas=[], confidence_threshold=bad)  # type: ignore[arg-type]

    def test_bounds_are_accepted(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.9), LABELS)
        for ok in (0, 0.0, 0.5, 1, 1.0):
            with self.subTest(threshold=ok):
                Jury(classifier=classifier, personas=[], confidence_threshold=ok)


class _JudgeRaisesClient(FakeLLMClient):
    """Persona calls succeed; any call to the judge model raises."""

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if model == "judge":
            self.calls.append({"model": model})
            raise RuntimeError("400 Bad Request")
        return await super().complete(
            model, system_prompt, prompt, temperature, response_format
        )


class JudgeFailureTests(unittest.IsolatedAsyncioTestCase):
    """BUG-03: a judge LLM failure after a paid debate must not reject classify()."""

    async def test_classify_returns_vote_when_judge_call_fails(self) -> None:
        llm = _JudgeRaisesClient({"persona": _persona_reply("unsafe", 0.9)})
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", 0.4), LABELS),
            personas=_personas(),
            judge=LLMJudge(model="judge", llm_client=llm),
            llm_client=llm,
            debate_config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=3),
        )

        verdict = await jury.classify("text")

        self.assertEqual(verdict.judge_strategy, "llm_judge_fallback_error")
        self.assertEqual(verdict.label, "unsafe")
        self.assertTrue(verdict.was_escalated)
        self.assertIsNotNone(verdict.debate_transcript)
        # 3 persona calls (unanimous opening round) at 0.001 each.
        self.assertAlmostEqual(verdict.total_cost_usd, 0.003)


if __name__ == "__main__":
    unittest.main()
