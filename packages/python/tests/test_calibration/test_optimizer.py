from __future__ import annotations

import json
import math
import unittest

from llm_jury.calibration.optimizer import ThresholdCalibrator
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona
from tests.helpers import FakeLLMClient, FakeLLMReply
from tests.test_evaluation.fixture import (
    EXPECTED,
    SWEEP_THRESHOLDS,
    TEXTS,
    CountingClassifier,
    ScriptedJury,
)

LABELS = ["safe", "unsafe"]


class ThresholdCalibratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_calibration_returns_threshold_in_range(self) -> None:
        confidences = {
            "a": ("safe", 0.9),
            "b": ("unsafe", 0.45),
            "c": ("safe", 0.65),
            "d": ("unsafe", 0.55),
        }
        classifier = FunctionClassifier(
            lambda text: confidences[text], ["safe", "unsafe"]
        )
        jury = Jury(classifier=classifier, personas=[], confidence_threshold=0.7)
        calibrator = ThresholdCalibrator(jury)

        threshold = await calibrator.calibrate(
            texts=["a", "b", "c", "d"],
            labels=["safe", "unsafe", "safe", "unsafe"],
            error_cost=10.0,
            escalation_cost=0.05,
            thresholds=[0.5, 0.6, 0.7, 0.8],
        )

        self.assertIn(threshold, [0.5, 0.6, 0.7, 0.8])
        report = calibrator.calibration_report()
        self.assertIn("best_threshold", report)
        self.assertEqual(report["best_threshold"], threshold)
        self.assertFalse(report["use_jury"])
        self.assertNotIn("summary", report)
        self.assertEqual(
            set(report["rows"][0]),
            {"threshold", "accuracy", "escalation_rate", "total_cost"},
        )


class CheapModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_primary_classifier_runs_once_per_text(self) -> None:
        classifier = CountingClassifier({"a": ("safe", 0.9), "b": ("unsafe", 0.4)})
        jury = Jury(classifier=classifier, personas=[], llm_client=FakeLLMClient())
        await ThresholdCalibrator(jury).calibrate(
            ["a", "b"], ["safe", "unsafe"], thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.assertEqual(classifier.calls, {"a": 1, "b": 1})

    async def test_escalated_items_are_left_out_of_accuracy(self) -> None:
        # Always wrong primary: escalations must not count as correct.
        classifier = CountingClassifier(
            {"a": ("unsafe", 0.4), "b": ("safe", 0.6), "c": ("safe", 0.9)}
        )
        jury = Jury(classifier=classifier, personas=[], llm_client=FakeLLMClient())
        calibrator = ThresholdCalibrator(jury)
        await calibrator.calibrate(
            ["a", "b", "c"],
            ["safe", "unsafe", "unsafe"],
            escalation_cost=0.05,
            thresholds=[0.5, 0.95],
        )
        rows = calibrator.calibration_report()["rows"]
        # t=0.5: a escalates, b and c are kept and both wrong.
        self.assertEqual(rows[0]["accuracy"], 0.0)
        self.assertAlmostEqual(rows[0]["escalation_rate"], 1 / 3)
        self.assertAlmostEqual(rows[0]["total_cost"], 20.05)
        # t=0.95: everything escalates, nothing is resolved.
        self.assertEqual(rows[1]["accuracy"], 0.0)
        self.assertEqual(rows[1]["escalation_rate"], 1.0)
        self.assertAlmostEqual(rows[1]["total_cost"], 0.15)

    async def test_non_finite_confidence_escalates_like_the_jury(self) -> None:
        classifier = CountingClassifier({"a": ("safe", math.nan), "b": ("safe", 0.9)})
        jury = Jury(classifier=classifier, personas=[], llm_client=FakeLLMClient())
        calibrator = ThresholdCalibrator(jury)
        await calibrator.calibrate(
            ["a", "b"], ["safe", "safe"], escalation_cost=0.05, thresholds=[0.5]
        )
        row = calibrator.calibration_report()["rows"][0]
        self.assertEqual(row["escalation_rate"], 0.5)
        self.assertEqual(row["accuracy"], 1.0)
        self.assertAlmostEqual(row["total_cost"], 0.05)

    async def test_escalation_cost_defaults_to_five_cents(self) -> None:
        classifier = CountingClassifier({"a": ("safe", 0.4)})
        jury = Jury(classifier=classifier, personas=[], llm_client=FakeLLMClient())
        calibrator = ThresholdCalibrator(jury)
        await calibrator.calibrate(["a"], ["safe"], thresholds=[0.5])
        self.assertAlmostEqual(
            calibrator.calibration_report()["rows"][0]["total_cost"], 0.05
        )


class JuryModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_rows_come_from_the_measured_sweep(self) -> None:
        jury = ScriptedJury()
        calibrator = ThresholdCalibrator(jury)  # type: ignore[arg-type]
        best = await calibrator.calibrate(
            TEXTS, EXPECTED, thresholds=SWEEP_THRESHOLDS, use_jury=True
        )

        self.assertEqual(best, 0.95)
        self.assertEqual(jury.threshold, 0.95)  # type: ignore[attr-defined]
        report = calibrator.calibration_report()
        self.assertTrue(report["use_jury"])
        self.assertEqual(report["best_threshold"], 0.95)
        self.assertEqual(report["summary"]["flips_helped"], 4)

        first = report["rows"][0]
        self.assertEqual(first["threshold"], 0.5)
        self.assertAlmostEqual(first["accuracy"], 0.5)  # primary, kept items only
        self.assertAlmostEqual(first["system_accuracy"], 0.6)
        self.assertAlmostEqual(first["jury_accuracy"], 1.0)
        self.assertAlmostEqual(first["escalation_rate"], 0.2)
        self.assertAlmostEqual(first["total_cost"], 40.5)
        last = report["rows"][-1]
        self.assertAlmostEqual(last["system_accuracy"], 0.7)
        self.assertAlmostEqual(last["total_cost"], 32.0)

        assert calibrator.evaluation_report is not None
        self.assertEqual(calibrator.evaluation_report.band_upper, 0.95)
        self.assertEqual(json.loads(json.dumps(report))["use_jury"], True)

    async def test_primary_classifier_runs_once_per_text(self) -> None:
        jury = ScriptedJury()
        await ThresholdCalibrator(jury).calibrate(  # type: ignore[arg-type]
            TEXTS, EXPECTED, use_jury=True
        )
        self.assertEqual(jury.classifier.calls, {text: 1 for text in TEXTS})

    async def test_debates_only_below_the_highest_threshold(self) -> None:
        jury = ScriptedJury()
        await ThresholdCalibrator(jury).calibrate(  # type: ignore[arg-type]
            TEXTS, EXPECTED, thresholds=[0.5, 0.6], use_jury=True
        )
        self.assertEqual(sorted(jury.escalated), ["t10", "t8", "t9"])

    async def test_runs_the_real_jury_with_a_fake_client(self) -> None:
        reply = FakeLLMReply(
            json.dumps(
                {
                    "label": "unsafe",
                    "confidence": 0.9,
                    "reasoning": "r",
                    "key_factors": [],
                }
            ),
            cost_usd=0.001,
        )
        client = FakeLLMClient({"persona": reply})
        classifier = CountingClassifier({"a": ("safe", 0.6), "b": ("safe", 0.99)})
        jury = Jury(
            classifier=classifier,
            personas=[Persona(name="A", role="r", system_prompt="A", model="persona")],
            judge=MajorityVoteJudge(),
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
            llm_client=client,
        )
        calibrator = ThresholdCalibrator(jury)
        best = await calibrator.calibrate(
            ["a", "b"], ["unsafe", "safe"], thresholds=[0.5, 0.7], use_jury=True
        )
        # Item a is wrong at 0.5 and fixed by the jury at 0.7.
        self.assertEqual(best, 0.7)
        self.assertEqual(classifier.calls, {"a": 1, "b": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(jury.stats.total, 0)


if __name__ == "__main__":
    unittest.main()
