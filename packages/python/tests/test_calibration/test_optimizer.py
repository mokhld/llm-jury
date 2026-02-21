from __future__ import annotations

import unittest

from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.calibration.optimizer import ThresholdCalibrator
from llm_jury.jury.core import Jury


class ThresholdCalibratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_calibration_returns_threshold_in_range(self) -> None:
        confidences = {
            "a": ("safe", 0.9),
            "b": ("unsafe", 0.45),
            "c": ("safe", 0.65),
            "d": ("unsafe", 0.55),
        }
        classifier = FunctionClassifier(lambda text: confidences[text], ["safe", "unsafe"])
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


if __name__ == "__main__":
    unittest.main()
