from __future__ import annotations

import json
import unittest
from typing import Any

from llm_jury.classifiers.base import ClassificationResult, Classifier
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode, DebateTranscript
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.llm.cache import CachingLLMClient
from llm_jury.personas.base import Persona, PersonaResponse
from tests.helpers import FakeLLMClient, FakeLLMReply

LABELS = ["safe", "unsafe"]


class _PricedClassifier(Classifier):
    def __init__(self, confidence: float, cost_usd: float | None) -> None:
        self.labels = LABELS
        self.confidence = confidence
        self.cost_usd = cost_usd

    async def classify(self, text: str) -> ClassificationResult:
        return ClassificationResult("safe", self.confidence, cost_usd=self.cost_usd)


def _persona_reply(cost: float | None = 0.001) -> FakeLLMReply:
    return FakeLLMReply(
        json.dumps(
            {
                "label": "unsafe",
                "confidence": 0.9,
                "reasoning": "r",
                "key_factors": [],
            }
        ),
        cost_usd=cost,
    )


def _judge_reply(cost: float | None = 0.004) -> FakeLLMReply:
    return FakeLLMReply(
        json.dumps(
            {
                "label": "unsafe",
                "confidence": 0.8,
                "reasoning": "judge",
                "key_agreements": [],
                "key_disagreements": [],
                "decisive_factor": "d",
            }
        ),
        cost_usd=cost,
    )


def _personas(count: int = 2) -> list[Persona]:
    return [
        Persona(name=f"P{i}", role="r", system_prompt=f"persona-{i}", model="persona")
        for i in range(count)
    ]


class _UnpricedDebateEngine:
    """Returns a debate whose calls reported no cost at all."""

    def __init__(self, unpriced_calls: int) -> None:
        self.unpriced_calls = unpriced_calls
        self.kwargs: dict[str, Any] = {}

    async def debate(
        self,
        text: str,
        primary_result: ClassificationResult,
        labels: list[str],
        max_cost_usd: float | None = None,
        estimated_cost_per_call_usd: float | None = None,
    ) -> DebateTranscript:
        self.kwargs = {
            "max_cost_usd": max_cost_usd,
            "estimated_cost_per_call_usd": estimated_cost_per_call_usd,
        }
        return DebateTranscript(
            input_text=text,
            primary_result=primary_result,
            rounds=[[PersonaResponse("P0", "unsafe", 0.9, "r", [])]],
            duration_ms=1,
            total_tokens=0,
            total_cost_usd=None,
            unpriced_calls=self.unpriced_calls,
        )


class EscalatedVerdictCostTests(unittest.IsolatedAsyncioTestCase):
    async def test_judged_verdict_includes_primary_debate_and_judge_cost(self) -> None:
        llm = FakeLLMClient({"persona": _persona_reply(0.001), "judge": _judge_reply()})
        jury = Jury(
            classifier=_PricedClassifier(0.4, cost_usd=0.002),
            personas=_personas(2),
            judge=LLMJudge(model="judge", llm_client=llm),
            llm_client=llm,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        verdict = await jury.classify("text")

        # primary 0.002 + 2 personas x 0.001 + judge 0.004
        self.assertAlmostEqual(verdict.total_cost_usd, 0.008)

    async def test_unpriced_everything_reports_unknown_cost(self) -> None:
        llm = FakeLLMClient({"persona": _persona_reply(None)})
        jury = Jury(
            classifier=_PricedClassifier(0.4, cost_usd=None),
            personas=_personas(2),
            judge=MajorityVoteJudge(),
            llm_client=llm,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        verdict = await jury.classify("text")

        self.assertIsNone(verdict.total_cost_usd)
        self.assertIsNone(verdict.debate_transcript.total_cost_usd)
        self.assertEqual(verdict.debate_transcript.unpriced_calls, 2)
        self.assertEqual(verdict.to_dict()["debate_transcript"]["unpriced_calls"], 2)

    async def test_priced_primary_with_unpriced_debate_reports_unknown(self) -> None:
        llm = FakeLLMClient({"persona": _persona_reply(None)})
        jury = Jury(
            classifier=_PricedClassifier(0.4, cost_usd=0.0),
            personas=_personas(2),
            judge=MajorityVoteJudge(),
            llm_client=llm,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        verdict = await jury.classify("text")

        # A free primary must not turn an unpriced debate into a $0 verdict.
        self.assertIsNone(verdict.total_cost_usd)

    async def test_fast_path_keeps_reported_primary_cost(self) -> None:
        jury = Jury(classifier=_PricedClassifier(0.9, cost_usd=None), personas=[])
        verdict = await jury.classify("text")
        self.assertIsNone(verdict.total_cost_usd)

        jury = Jury(classifier=_PricedClassifier(0.9, cost_usd=0.002), personas=[])
        verdict = await jury.classify("text")
        self.assertAlmostEqual(verdict.total_cost_usd, 0.002)

    async def test_user_override_skip_reports_primary_cost(self) -> None:
        jury = Jury(
            classifier=_PricedClassifier(0.4, cost_usd=0.002),
            personas=_personas(1),
            judge=MajorityVoteJudge(),
            llm_client=FakeLLMClient(),
            on_cost_estimate=lambda _e, _t: False,
        )
        verdict = await jury.classify("text")
        self.assertEqual(verdict.judge_strategy, "cost_guard_user_override")
        self.assertAlmostEqual(verdict.total_cost_usd, 0.002)

    async def test_pre_flight_skip_reports_primary_cost(self) -> None:
        jury = Jury(
            classifier=_PricedClassifier(0.4, cost_usd=0.002),
            personas=_personas(3),
            judge=MajorityVoteJudge(),
            llm_client=FakeLLMClient(),
            max_debate_cost_usd=0.001,
        )
        verdict = await jury.classify("text")
        self.assertEqual(verdict.judge_strategy, "cost_guard_pre_flight")
        self.assertAlmostEqual(verdict.total_cost_usd, 0.002)

    async def test_primary_fallback_reports_primary_and_debate_cost(self) -> None:
        llm = FakeLLMClient({"persona": _persona_reply(0.5)})
        jury = Jury(
            classifier=_PricedClassifier(0.4, cost_usd=0.002),
            personas=_personas(2),
            judge=MajorityVoteJudge(),
            llm_client=llm,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
            max_debate_cost_usd=0.9,
        )
        verdict = await jury.classify("text")
        self.assertEqual(verdict.judge_strategy, "cost_guard_primary_fallback")
        self.assertAlmostEqual(verdict.total_cost_usd, 1.002)


class CostGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_unpriced_debate_trips_post_debate_guard(self) -> None:
        engine = _UnpricedDebateEngine(unpriced_calls=5)
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", 0.4), LABELS),
            personas=_personas(1),
            judge=MajorityVoteJudge(),
            llm_client=FakeLLMClient(),
            max_debate_cost_usd=0.03,
            estimated_cost_per_persona_usd=0.01,
        )
        jury.debate_engine = engine

        verdict = await jury.classify("text")

        # 5 unpriced calls x 0.01 = 0.05 > 0.03, although no cost was reported.
        self.assertEqual(verdict.judge_strategy, "cost_guard_primary_fallback")
        self.assertEqual(
            engine.kwargs,
            {"max_cost_usd": 0.03, "estimated_cost_per_call_usd": 0.01},
        )

    async def test_unpriced_debate_within_cap_is_judged(self) -> None:
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", 0.4), LABELS),
            personas=_personas(1),
            judge=MajorityVoteJudge(),
            llm_client=FakeLLMClient(),
            max_debate_cost_usd=0.03,
            estimated_cost_per_persona_usd=0.01,
        )
        jury.debate_engine = _UnpricedDebateEngine(unpriced_calls=3)

        verdict = await jury.classify("text")

        # Exactly at the cap (0.03) does not trip.
        self.assertEqual(verdict.judge_strategy, "majority_vote")

    def test_estimate_counts_every_possible_call(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.9), LABELS)
        cases = [
            # (mode, max_rounds, judge, expected calls)
            (
                DebateMode.DELIBERATION,
                3,
                LLMJudge(llm_client=FakeLLMClient()),
                4 * 3 + 2,
            ),
            (DebateMode.DELIBERATION, 0, MajorityVoteJudge(), 4 * 1 + 1),
            (DebateMode.INDEPENDENT, 3, LLMJudge(llm_client=FakeLLMClient()), 4 + 1),
            (DebateMode.SEQUENTIAL, 3, MajorityVoteJudge(), 4),
            (DebateMode.ADVERSARIAL, 3, MajorityVoteJudge(), 4),
        ]
        for mode, max_rounds, judge, calls in cases:
            with self.subTest(mode=mode, judge=type(judge).__name__):
                jury = Jury(
                    classifier=classifier,
                    personas=_personas(4),
                    judge=judge,
                    llm_client=FakeLLMClient(),
                    debate_config=DebateConfig(mode=mode, max_rounds=max_rounds),
                    estimated_cost_per_persona_usd=0.01,
                )
                self.assertAlmostEqual(jury.estimated_max_debate_cost_usd, 0.01 * calls)

    async def test_estimate_includes_summariser_and_judge_for_pre_flight(self) -> None:
        # 3 personas x 2 rounds = 0.06 used to pass a 0.07 cap; the summariser
        # and judge calls bring the real maximum to 0.08.
        llm = FakeLLMClient()
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", 0.4), LABELS),
            personas=_personas(3),
            llm_client=llm,
            max_debate_cost_usd=0.07,
            estimated_cost_per_persona_usd=0.01,
        )

        verdict = await jury.classify("text")

        self.assertEqual(verdict.judge_strategy, "cost_guard_pre_flight")
        self.assertEqual(llm.calls, [])


class CachedCallCostTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_hits_are_free(self) -> None:
        inner = FakeLLMClient({"persona": _persona_reply(0.001)})
        cache = CachingLLMClient(inner)
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", 0.4), LABELS),
            personas=_personas(1),
            judge=MajorityVoteJudge(),
            llm_client=cache,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        first = await jury.classify("same text")
        second = await jury.classify("same text")

        self.assertAlmostEqual(first.total_cost_usd, 0.001)
        self.assertEqual(second.total_cost_usd, 0.0)
        self.assertEqual(len(inner.calls), 1)


if __name__ == "__main__":
    unittest.main()
