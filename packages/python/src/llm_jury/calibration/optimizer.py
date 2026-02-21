from __future__ import annotations

from dataclasses import dataclass

from llm_jury.jury.core import Jury


@dataclass(slots=True)
class _CalibrationRow:
    threshold: float
    accuracy: float
    escalation_rate: float
    total_cost: float


class ThresholdCalibrator:
    """Finds the optimal confidence threshold for escalation.

    Classifies each text **once**, caches the results, and then sweeps
    threshold candidates over the cached ``(label, confidence)`` pairs.
    """

    def __init__(self, jury: Jury) -> None:
        self.jury = jury
        self._rows: list[_CalibrationRow] = []
        self._best_threshold: float | None = None

    async def calibrate(
        self,
        texts: list[str],
        labels: list[str],
        error_cost: float = 10.0,
        escalation_cost: float = 0.05,
        thresholds: list[float] | None = None,
    ) -> float:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")

        candidates = thresholds or [round(x / 100.0, 2) for x in range(50, 100, 5)]
        if not candidates:
            raise ValueError("at least one threshold is required")

        # Classify each text once and cache the results.
        cached_results = []
        for text in texts:
            result = await self.jury.classifier.classify(text)
            cached_results.append(result)

        self._rows = []
        best_threshold = candidates[0]
        best_cost = float("inf")

        for threshold in candidates:
            errors = 0
            escalations = 0
            correct = 0

            for result, expected in zip(cached_results, labels):
                if result.confidence < threshold:
                    # Item would be escalated — we don't know the jury's
                    # verdict without running it, so treat as unresolved.
                    escalations += 1
                else:
                    if result.label == expected:
                        correct += 1
                    else:
                        errors += 1

            total = max(1, len(texts))
            resolved = correct + errors
            accuracy = correct / resolved if resolved > 0 else 0.0
            total_cost = errors * error_cost + escalations * escalation_cost
            row = _CalibrationRow(
                threshold=threshold,
                accuracy=accuracy,
                escalation_rate=escalations / total,
                total_cost=total_cost,
            )
            self._rows.append(row)

            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold

        self._best_threshold = best_threshold
        self.jury.threshold = best_threshold
        return best_threshold

    def calibration_report(self) -> dict:
        return {
            "best_threshold": self._best_threshold,
            "rows": [
                {
                    "threshold": row.threshold,
                    "accuracy": row.accuracy,
                    "escalation_rate": row.escalation_rate,
                    "total_cost": row.total_cost,
                }
                for row in self._rows
            ],
        }
