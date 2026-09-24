from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_jury.evaluation.evaluator import (
    DEFAULT_ESCALATION_COST_USD,
    DEFAULT_THRESHOLDS,
    EvaluationReport,
    JuryEvaluator,
)
from llm_jury.jury.core import Jury
from llm_jury.utils import is_finite_number


@dataclass(slots=True)
class _CalibrationRow:
    threshold: float
    # Primary accuracy on the items that do not escalate (0.0 when all do).
    accuracy: float
    escalation_rate: float
    total_cost: float
    # Only measured with use_jury=True.
    system_accuracy: float | None = None
    jury_accuracy: float | None = None


class ThresholdCalibrator:
    """Finds the confidence threshold with the lowest expected cost.

    Classifies each text **once**, then sweeps threshold candidates over the
    cached ``(label, confidence)`` pairs. An item escalates at threshold ``t``
    when its confidence is below ``t`` or not finite, the same rule ``Jury``
    uses.

    ``use_jury=False`` (the default) never runs the jury: escalations are
    priced at ``escalation_cost`` and left out of ``accuracy``, and nothing is
    assumed about whether the jury would get them right. ``use_jury=True``
    runs :class:`~llm_jury.evaluation.JuryEvaluator` (one debate per item below
    the highest threshold) and picks the threshold from the measured
    outcomes.
    """

    def __init__(self, jury: Jury) -> None:
        self.jury = jury
        self._rows: list[_CalibrationRow] = []
        self._best_threshold: float | None = None
        self._use_jury = False
        # The jury measurement behind the last use_jury=True calibration.
        self.evaluation_report: EvaluationReport | None = None

    async def calibrate(
        self,
        texts: list[str],
        labels: list[str],
        error_cost: float = 10.0,
        escalation_cost: float | None = None,
        thresholds: list[float] | None = None,
        use_jury: bool = False,
    ) -> float:
        """Pick the threshold that minimises ``errors * error_cost +
        escalations * escalation_cost`` and set it on the jury.

        ``escalation_cost=None`` means 0.05 in the default mode and, with
        ``use_jury=True``, the measured mean debate cost (0.05 when no debate
        reported a cost). With ``use_jury=True`` errors count the jury's
        label for escalated items; otherwise escalated items count no error.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")

        candidates = thresholds or list(DEFAULT_THRESHOLDS)
        if not candidates:
            raise ValueError("at least one threshold is required")

        if use_jury:
            rows = await self._measured_rows(
                texts, labels, candidates, error_cost, escalation_cost
            )
        else:
            rows = await self._cheap_rows(
                texts,
                labels,
                candidates,
                error_cost,
                (
                    DEFAULT_ESCALATION_COST_USD
                    if escalation_cost is None
                    else escalation_cost
                ),
            )

        best = rows[0]
        for row in rows[1:]:
            if row.total_cost < best.total_cost:
                best = row

        self._rows = rows
        self._use_jury = use_jury
        self._best_threshold = best.threshold
        self.jury.threshold = best.threshold
        return best.threshold

    async def _cheap_rows(
        self,
        texts: list[str],
        labels: list[str],
        candidates: list[float],
        error_cost: float,
        escalation_cost: float,
    ) -> list[_CalibrationRow]:
        self.evaluation_report = None
        # Classify each text once and cache the results.
        cached_results = []
        for text in texts:
            result = await self.jury.classifier.classify(text)
            cached_results.append(result)

        rows: list[_CalibrationRow] = []
        for threshold in candidates:
            errors = 0
            escalations = 0
            correct = 0

            for result, expected in zip(cached_results, labels, strict=True):
                if (
                    not is_finite_number(result.confidence)
                    or result.confidence < threshold
                ):
                    # The item would be escalated. Without running the jury its
                    # outcome is unknown, so it counts as neither right nor wrong.
                    escalations += 1
                elif result.label == expected:
                    correct += 1
                else:
                    errors += 1

            total = max(1, len(texts))
            resolved = correct + errors
            rows.append(
                _CalibrationRow(
                    threshold=threshold,
                    accuracy=correct / resolved if resolved > 0 else 0.0,
                    escalation_rate=escalations / total,
                    total_cost=errors * error_cost + escalations * escalation_cost,
                )
            )
        return rows

    async def _measured_rows(
        self,
        texts: list[str],
        labels: list[str],
        candidates: list[float],
        error_cost: float,
        escalation_cost: float | None,
    ) -> list[_CalibrationRow]:
        report = await JuryEvaluator(self.jury).evaluate(
            texts, labels, band_upper=max(candidates)
        )
        self.evaluation_report = report
        sweep = report.threshold_sweep(
            candidates, error_cost=error_cost, escalation_cost=escalation_cost
        )
        return [
            _CalibrationRow(
                threshold=row["threshold"],
                accuracy=row["primary_accuracy"] or 0.0,
                escalation_rate=row["escalation_rate"] or 0.0,
                total_cost=row["total_cost"],
                system_accuracy=row["system_accuracy"],
                jury_accuracy=row["jury_accuracy"],
            )
            for row in sweep
        ]

    def calibration_report(self) -> dict[str, Any]:
        """The last calibration: best threshold and one row per candidate.

        Rows carry ``threshold``, ``accuracy`` (primary accuracy on the items
        that do not escalate), ``escalation_rate`` and ``total_cost``. After
        ``use_jury=True`` they also carry ``system_accuracy`` and
        ``jury_accuracy``, and the report adds the evaluation ``summary``.
        """
        rows: list[dict[str, Any]] = []
        for row in self._rows:
            data: dict[str, Any] = {
                "threshold": row.threshold,
                "accuracy": row.accuracy,
                "escalation_rate": row.escalation_rate,
                "total_cost": row.total_cost,
            }
            if self._use_jury:
                data["system_accuracy"] = row.system_accuracy
                data["jury_accuracy"] = row.jury_accuracy
            rows.append(data)

        report: dict[str, Any] = {
            "best_threshold": self._best_threshold,
            "use_jury": self._use_jury,
            "rows": rows,
        }
        if self._use_jury and self.evaluation_report is not None:
            report["summary"] = self.evaluation_report.summary()
        return report
