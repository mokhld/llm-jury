from __future__ import annotations

import json
import math
import unittest

from llm_jury.classifiers.base import ClassificationResult, Classifier
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.evaluation import (
    EvaluationReport,
    JuryEvaluator,
    TooManyEscalationsError,
)
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona
from tests.helpers import FakeLLMClient, FakeLLMReply
from tests.test_evaluation.fixture import (
    EXPECTED,
    FIXTURE,
    LABELS,
    SWEEP_THRESHOLDS,
    TEXTS,
    CountingClassifier,
    ScriptedJury,
)


async def _evaluate(jury: ScriptedJury, **kwargs) -> EvaluationReport:
    return await JuryEvaluator(jury).evaluate(TEXTS, EXPECTED, **kwargs)


class EvaluatorFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_matches_hand_computed_values(self) -> None:
        report = await _evaluate(ScriptedJury())
        summary = report.summary()

        self.assertEqual(summary["n"], 10)
        self.assertEqual(summary["band_upper"], 0.95)
        self.assertAlmostEqual(summary["primary_accuracy"], 0.4)
        self.assertEqual(summary["debated"], 8)
        self.assertAlmostEqual(summary["jury_accuracy_on_debated"], 0.75)
        self.assertAlmostEqual(summary["primary_accuracy_on_debated"], 0.375)
        self.assertEqual(summary["flips_helped"], 4)
        self.assertEqual(summary["flips_hurt"], 1)
        self.assertAlmostEqual(summary["debate_cost_usd"], 1.75)
        self.assertEqual(summary["unpriced_calls"], 3)
        self.assertAlmostEqual(summary["mean_debate_cost_usd"], 0.25)
        self.assertEqual(summary["latency_ms_p50"], 400)
        self.assertEqual(summary["latency_ms_p95"], 800)
        self.assertEqual(summary["degraded"], 2)
        self.assertEqual(
            summary["fallbacks"],
            {
                "llm_judge_fallback_error": 1,
                "llm_judge_fallback_personas_failed": 1,
            },
        )
        self.assertEqual(
            summary["confusion"],
            {
                "primary": {
                    "safe": {"safe": 2, "unsafe": 3},
                    "unsafe": {"safe": 3, "unsafe": 2},
                },
                "jury": {
                    "safe": {"safe": 3, "unsafe": 1},
                    "unsafe": {"safe": 1, "unsafe": 3},
                },
            },
        )

    async def test_items_record_primary_and_jury_outcomes(self) -> None:
        report = await _evaluate(ScriptedJury())
        by_text = {item.text: item for item in report.items}

        top = by_text["t1"]
        self.assertFalse(top.debated)
        self.assertIsNone(top.jury_label)
        self.assertTrue(top.primary_correct)
        self.assertEqual(top.primary_cost_usd, 0.0)

        helped = by_text["t3"]
        self.assertTrue(helped.debated)
        self.assertFalse(helped.primary_correct)
        self.assertTrue(helped.jury_correct)
        self.assertEqual(helped.jury_label, "unsafe")
        self.assertEqual(helped.jury_cost_usd, 0.25)
        self.assertEqual(helped.jury_duration_ms, 100)
        self.assertEqual(helped.jury_strategy, "llm_judge")

        unpriced = by_text["t7"]
        self.assertIsNone(unpriced.jury_cost_usd)
        self.assertEqual(unpriced.unpriced_calls, 3)
        self.assertTrue(unpriced.jury_degraded)

    async def test_threshold_sweep_matches_hand_computed_values(self) -> None:
        report = await _evaluate(ScriptedJury())
        rows = report.threshold_sweep(SWEEP_THRESHOLDS, error_cost=10.0)

        # (threshold, escalation_rate, system_accuracy, jury_accuracy,
        #  primary_accuracy, errors, total_cost); escalation cost is the
        # measured mean debate cost, 0.25.
        expected = [
            (0.5, 0.2, 0.6, 1.0, 4 / 8, 4, 40.5),
            (0.6, 0.3, 0.6, 1.0, 3 / 7, 4, 40.75),
            (0.7, 0.4, 0.6, 3 / 4, 3 / 6, 4, 41.0),
            (0.8, 0.6, 0.6, 4 / 6, 2 / 4, 4, 41.5),
            (0.9, 0.7, 0.6, 5 / 7, 1 / 3, 4, 41.75),
            (0.95, 0.8, 0.7, 6 / 8, 1 / 2, 3, 32.0),
        ]
        self.assertEqual(len(rows), len(expected))
        for row, want in zip(rows, expected, strict=True):
            with self.subTest(threshold=want[0]):
                self.assertEqual(row["threshold"], want[0])
                self.assertAlmostEqual(row["escalation_rate"], want[1])
                self.assertAlmostEqual(row["system_accuracy"], want[2])
                self.assertAlmostEqual(row["jury_accuracy"], want[3])
                self.assertAlmostEqual(row["primary_accuracy"], want[4])
                self.assertEqual(row["errors"], want[5])
                self.assertAlmostEqual(row["total_cost"], want[6])

    async def test_best_threshold_matches_hand_computed_values(self) -> None:
        report = await _evaluate(ScriptedJury())
        self.assertEqual(
            report.best_threshold(error_cost=10.0, thresholds=SWEEP_THRESHOLDS), 0.95
        )
        # Errors cheap, escalations expensive: escalate as little as possible.
        self.assertEqual(
            report.best_threshold(
                error_cost=1.0, escalation_cost=1.0, thresholds=SWEEP_THRESHOLDS
            ),
            0.5,
        )

    async def test_always_wrong_jury_picks_the_lowest_threshold(self) -> None:
        report = await _evaluate(ScriptedJury(mode="wrong"))
        self.assertEqual(report.best_threshold(escalation_cost=0.01), 0.5)
        self.assertEqual(report.best_threshold(), 0.5)

    async def test_always_right_jury_with_cheap_escalations_picks_the_highest(
        self,
    ) -> None:
        report = await _evaluate(ScriptedJury(mode="right"))
        self.assertEqual(report.best_threshold(escalation_cost=0.01), 0.95)

    async def test_to_dict_is_json_serialisable(self) -> None:
        report = await _evaluate(ScriptedJury())
        data = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(data["band_upper"], 0.95)
        self.assertEqual(data["summary"]["flips_helped"], 4)
        self.assertEqual(len(data["items"]), 10)
        self.assertEqual(data["items"][6]["jury_cost_usd"], None)


class EvaluatorRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_primary_classifier_runs_exactly_once_per_text(self) -> None:
        jury = ScriptedJury()
        await _evaluate(jury)
        self.assertEqual(jury.classifier.calls, {text: 1 for text in TEXTS})

    async def test_only_items_below_band_upper_are_debated(self) -> None:
        jury = ScriptedJury()
        await _evaluate(jury, band_upper=0.7)
        # Confidences below 0.7: t7 (0.60), t8, t9, t10. t6 is exactly 0.70.
        self.assertEqual(sorted(jury.escalated), ["t10", "t7", "t8", "t9"])

    async def test_max_escalations_raises_before_any_debate(self) -> None:
        jury = ScriptedJury()
        with self.assertRaisesRegex(TooManyEscalationsError, "8 item"):
            await _evaluate(jury, max_escalations=7)
        self.assertEqual(jury.escalated, [])
        self.assertTrue(issubclass(TooManyEscalationsError, ValueError))

        # Exactly at the cap is allowed.
        report = await _evaluate(ScriptedJury(), max_escalations=8)
        self.assertEqual(report.summary()["debated"], 8)

    async def test_max_escalations_raises_before_any_llm_call(self) -> None:
        client = FakeLLMClient()
        jury = _llm_jury(client, confidence=0.5)
        with self.assertRaises(TooManyEscalationsError):
            await JuryEvaluator(jury).evaluate(["a", "b"], LABELS, max_escalations=1)
        self.assertEqual(client.calls, [])

    async def test_non_finite_confidence_is_always_debated(self) -> None:
        rows = [
            ("n1", "safe", "unsafe", math.nan, "safe", 0.25, 0, 0, "llm_judge", 10),
            ("n2", "safe", "safe", 0.99, None, None, 0, 0, None, None),
        ]
        jury = ScriptedJury(rows=rows)
        report = await JuryEvaluator(jury).evaluate(["n1", "n2"], ["safe", "safe"])
        self.assertEqual(jury.escalated, ["n1"])
        row = report.threshold_sweep([0.5])[0]
        self.assertEqual(row["errors"], 0)
        self.assertEqual(row["escalation_rate"], 0.5)

    async def test_jury_stats_are_not_touched(self) -> None:
        jury = _llm_jury(FakeLLMClient(), confidence=0.5)
        await JuryEvaluator(jury).evaluate(["a", "b"], ["safe", "unsafe"])
        self.assertEqual(
            (jury.stats.total, jury.stats.fast_path, jury.stats.escalated), (0, 0, 0)
        )

    async def test_invalid_arguments_are_rejected(self) -> None:
        evaluator = JuryEvaluator(ScriptedJury())
        with self.assertRaisesRegex(ValueError, "same length"):
            await evaluator.evaluate(["t1"], [])
        for kwargs in (
            {"band_upper": 1.5},
            {"band_upper": math.nan},
            {"concurrency": 0},
            {"max_escalations": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    await evaluator.evaluate(TEXTS, EXPECTED, **kwargs)

    async def test_jury_without_personas_is_rejected_before_classifying(self) -> None:
        classifier = CountingClassifier({"a": ("safe", 0.5)})
        jury = Jury(classifier=classifier, personas=[], llm_client=FakeLLMClient())
        with self.assertRaisesRegex(ValueError, "at least one persona"):
            await JuryEvaluator(jury).evaluate(["a"], ["safe"])
        self.assertEqual(classifier.calls, {})


class ThresholdSweepRulesTests(unittest.IsolatedAsyncioTestCase):
    async def test_thresholds_above_band_upper_are_rejected(self) -> None:
        report = await _evaluate(ScriptedJury(), band_upper=0.8)
        with self.assertRaisesRegex(ValueError, "above band_upper"):
            report.threshold_sweep([0.5, 0.9])
        with self.assertRaisesRegex(ValueError, "above band_upper"):
            report.best_threshold(thresholds=[0.85])

    async def test_default_thresholds_stop_at_band_upper(self) -> None:
        report = await _evaluate(ScriptedJury(), band_upper=0.8)
        thresholds = [row["threshold"] for row in report.threshold_sweep()]
        self.assertEqual(thresholds, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    async def test_escalation_cost_defaults_to_measured_mean(self) -> None:
        report = await _evaluate(ScriptedJury())
        row = report.threshold_sweep([0.5], error_cost=0.0)[0]
        self.assertAlmostEqual(row["total_cost"], 2 * 0.25)

    async def test_escalation_cost_falls_back_when_nothing_was_priced(self) -> None:
        rows = [
            (r[0], r[1], r[2], r[3], r[4], None, 2, 0, r[8], r[9]) if r[4] else r
            for r in FIXTURE
        ]
        report = await _evaluate(ScriptedJury(rows=rows))
        self.assertIsNone(report.summary()["mean_debate_cost_usd"])
        row = report.threshold_sweep([0.5], error_cost=0.0)[0]
        self.assertAlmostEqual(row["total_cost"], 2 * 0.05)

    async def test_nothing_debated_gives_empty_jury_fields(self) -> None:
        report = await _evaluate(ScriptedJury(), band_upper=0.3)
        summary = report.summary()
        self.assertEqual(summary["debated"], 0)
        self.assertIsNone(summary["jury_accuracy_on_debated"])
        self.assertIsNone(summary["debate_cost_usd"])
        self.assertIsNone(summary["latency_ms_p50"])
        self.assertEqual(summary["fallbacks"], {})
        rows = report.threshold_sweep()
        self.assertEqual([row["threshold"] for row in rows], [0.3])
        self.assertIsNone(rows[0]["jury_accuracy"])


def _personas() -> list[Persona]:
    return [
        Persona(name=f"P{i}", role="r", system_prompt=f"P{i}", model="persona-model")
        for i in range(3)
    ]


def _llm_jury(client: FakeLLMClient, confidence: float) -> Jury:
    classifier = FunctionClassifier(lambda text: ("safe", confidence), LABELS)
    return Jury(
        classifier=classifier,
        personas=_personas(),
        judge=MajorityVoteJudge(),
        llm_client=client,
    )


class EvaluatorWithRealJuryTests(unittest.IsolatedAsyncioTestCase):
    def _reply(self, cost: float | None) -> FakeLLMReply:
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

    async def test_unknown_debate_cost_is_none_with_unpriced_calls(self) -> None:
        client = FakeLLMClient({"persona-model": self._reply(None)})
        jury = _llm_jury(client, confidence=0.5)
        report = await JuryEvaluator(jury).evaluate(["a", "b"], ["unsafe", "safe"])
        summary = report.summary()

        self.assertEqual(summary["debated"], 2)
        self.assertIsNone(summary["debate_cost_usd"])
        self.assertIsNone(summary["mean_debate_cost_usd"])
        self.assertEqual(summary["unpriced_calls"], 6)
        for item in report.items:
            self.assertIsNone(item.jury_cost_usd)
            self.assertEqual(item.unpriced_calls, 3)
        self.assertEqual(json.loads(json.dumps(summary))["debate_cost_usd"], None)

    async def test_known_debate_cost_excludes_the_primary_cost(self) -> None:
        client = FakeLLMClient({"persona-model": self._reply(0.002)})

        class PricedClassifier(Classifier):
            labels = LABELS

            async def classify(self, text: str) -> ClassificationResult:
                return ClassificationResult("safe", 0.5, cost_usd=0.5)

        jury = Jury(
            classifier=PricedClassifier(),
            personas=_personas(),
            judge=MajorityVoteJudge(),
            llm_client=client,
        )
        report = await JuryEvaluator(jury).evaluate(["a"], ["unsafe"])
        item = report.items[0]
        self.assertEqual(item.primary_cost_usd, 0.5)
        self.assertAlmostEqual(item.jury_cost_usd, 0.006)
        self.assertEqual(item.jury_label, "unsafe")
        self.assertTrue(item.jury_correct)
        self.assertEqual(report.summary()["flips_helped"], 1)


if __name__ == "__main__":
    unittest.main()
