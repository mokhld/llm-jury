"""Threshold calibration using labelled data.

Uses FunctionClassifier to demonstrate the calibration workflow
without requiring API calls. In production, swap in your real
classifier (LLMClassifier, HuggingFaceClassifier, etc.).
"""
from __future__ import annotations

import asyncio

from llm_jury import Jury
from llm_jury.classifiers import FunctionClassifier
from llm_jury.calibration.optimizer import ThresholdCalibrator


async def main() -> None:
    # Simulated classifier outputs for calibration data.
    predictions = {
        "a": ("safe", 0.9),
        "b": ("unsafe", 0.4),
        "c": ("safe", 0.65),
        "d": ("unsafe", 0.55),
    }

    classifier = FunctionClassifier(
        fn=lambda text: predictions[text],
        labels=["safe", "unsafe"],
    )

    jury = Jury(classifier=classifier, personas=[], confidence_threshold=0.7)
    calibrator = ThresholdCalibrator(jury)

    best = await calibrator.calibrate(
        texts=["a", "b", "c", "d"],
        labels=["safe", "unsafe", "safe", "unsafe"],
        thresholds=[0.5, 0.6, 0.7, 0.8],
    )

    print(f"Best threshold: {best}")
    print()

    report = calibrator.calibration_report()
    for row in report["rows"]:
        print(
            f"  threshold={row['threshold']:.2f}  "
            f"accuracy={row['accuracy']:.2f}  "
            f"escalation_rate={row['escalation_rate']:.2f}  "
            f"cost={row['total_cost']:.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
