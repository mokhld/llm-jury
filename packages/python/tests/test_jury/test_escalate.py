from __future__ import annotations

import json
import unittest
from typing import Any

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.judges.base import Verdict
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona
from tests.helpers import FakeLLMClient, FakeLLMReply

LABELS = ["safe", "unsafe"]


def _persona_reply(cost: float | None = 0.001) -> FakeLLMReply:
    return FakeLLMReply(
        json.dumps(
            {"label": "unsafe", "confidence": 0.9, "reasoning": "r", "key_factors": []}
        ),
        cost_usd=cost,
    )


class EscalateTests(unittest.IsolatedAsyncioTestCase):
    """Jury.escalate runs the escalation branch for an existing primary result."""

    def _jury(self, **kwargs: Any) -> tuple[Jury, FakeLLMClient, list]:
        seen: list[Any] = []
        client = kwargs.pop("llm_client", None) or FakeLLMClient(
            {"persona": _persona_reply()}
        )
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", 0.4), LABELS),
            personas=[
                Persona(name=n, role="r", system_prompt=f"persona-{n}", model="persona")
                for n in ("A", "B")
            ],
            judge=MajorityVoteJudge(),
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
            llm_client=client,
            on_escalation=lambda text, primary: seen.append(("escalation", text)),
            on_verdict=lambda verdict: seen.append(("verdict", verdict)),
            **kwargs,
        )
        return jury, client, seen

    async def test_debates_even_a_confident_primary(self) -> None:
        jury, client, _ = self._jury()
        primary = ClassificationResult("safe", 0.99, cost_usd=0.0)
        verdict = await jury.escalate("text", primary)

        self.assertTrue(verdict.was_escalated)
        self.assertEqual(verdict.judge_strategy, "majority_vote")
        self.assertEqual(verdict.label, "unsafe")
        self.assertIs(verdict.primary_result, primary)
        self.assertEqual(len(client.calls), 2)
        self.assertAlmostEqual(verdict.total_cost_usd, 0.002)

    async def test_does_not_touch_stats_but_fires_callbacks(self) -> None:
        jury, _, seen = self._jury()
        verdict = await jury.escalate("text", ClassificationResult("safe", 0.5))

        self.assertEqual(
            (jury.stats.total, jury.stats.fast_path, jury.stats.escalated), (0, 0, 0)
        )
        self.assertEqual(seen, [("escalation", "text"), ("verdict", verdict)])

    async def test_never_calls_the_primary_classifier(self) -> None:
        jury, _, _ = self._jury()
        calls: list[str] = []

        async def classify(text: str) -> ClassificationResult:
            calls.append(text)
            return ClassificationResult("safe", 0.4)

        jury.classifier.classify = classify  # type: ignore[method-assign]
        await jury.escalate("text", ClassificationResult("safe", 0.4))
        self.assertEqual(calls, [])

    async def test_cost_gates_apply(self) -> None:
        jury, client, _ = self._jury(on_cost_estimate=lambda _e, _t: False)
        verdict = await jury.escalate("text", ClassificationResult("safe", 0.4))
        self.assertEqual(verdict.judge_strategy, "cost_guard_user_override")

        jury, client, _ = self._jury(max_debate_cost_usd=0.0001)
        verdict = await jury.escalate("text", ClassificationResult("safe", 0.4))
        self.assertEqual(verdict.judge_strategy, "cost_guard_pre_flight")
        self.assertEqual(client.calls, [])

    async def test_matches_the_escalated_branch_of_classify(self) -> None:
        jury, _, _ = self._jury()
        via_classify = await jury.classify("text")
        primary = await jury.classifier.classify("text")
        via_escalate = await jury.escalate("text", primary)

        for field in ("label", "confidence", "judge_strategy", "total_cost_usd"):
            self.assertEqual(getattr(via_classify, field), getattr(via_escalate, field))
        self.assertEqual(via_classify.persona_failures, via_escalate.persona_failures)
        self.assertEqual(jury.stats.escalated, 1)

    async def test_classify_escalates_through_the_shared_branch(self) -> None:
        jury, _, _ = self._jury()
        seen: list[tuple[str, ClassificationResult]] = []
        original = jury._escalate

        async def spy(
            text: str, primary: ClassificationResult, start: float
        ) -> Verdict:
            seen.append((text, primary))
            return await original(text, primary, start)

        jury._escalate = spy  # type: ignore[method-assign]
        await jury.classify("low")
        await jury.escalate("direct", ClassificationResult("safe", 0.9))
        self.assertEqual([text for text, _ in seen], ["low", "direct"])

    async def test_requires_personas(self) -> None:
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", 0.4), LABELS),
            personas=[],
            llm_client=FakeLLMClient(),
        )
        with self.assertRaisesRegex(ValueError, "at least one persona"):
            await jury.escalate("text", ClassificationResult("safe", 0.4))


if __name__ == "__main__":
    unittest.main()
