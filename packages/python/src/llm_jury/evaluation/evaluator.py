"""Measure what the jury does on labelled data.

``JuryEvaluator`` classifies every text once with the jury's primary
classifier, sends the items whose confidence is below ``band_upper`` to
``Jury.escalate``, and returns an ``EvaluationReport``. The report compares the
jury's labels with the primary labels and sweeps escalation thresholds over the
measured outcomes, so the threshold is chosen from what the jury actually did.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Protocol, TypeVar

from llm_jury.classifiers.base import ClassificationResult, Classifier
from llm_jury.judges.base import Verdict
from llm_jury.utils import is_finite_number

T = TypeVar("T")

# Used by ``threshold_sweep`` when ``escalation_cost`` is None and no debate
# reported a cost.
DEFAULT_ESCALATION_COST_USD = 0.05

# The ThresholdCalibrator grid: 0.5, 0.55, ..., 0.95.
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(
    round(x / 100.0, 2) for x in range(50, 100, 5)
)


class TooManyEscalationsError(ValueError):
    """More items fall below ``band_upper`` than ``max_escalations`` allows."""


class EvaluableJury(Protocol):
    """What ``JuryEvaluator`` needs from a jury. ``Jury`` satisfies it."""

    classifier: Classifier

    async def escalate(self, text: str, primary: ClassificationResult) -> Verdict:
        """Debate and judge one item that already has a primary result."""
        ...


@dataclass(slots=True)
class EvaluationItem:
    """One labelled text: the primary result and, if debated, the jury's."""

    text: str
    expected: str
    primary_label: str
    primary_confidence: float
    primary_correct: bool
    primary_cost_usd: float | None
    debated: bool = False
    jury_label: str | None = None
    jury_confidence: float | None = None
    jury_correct: bool | None = None
    jury_strategy: str | None = None
    # Debate and judge cost: the verdict total minus the primary cost. None
    # when either is unknown.
    jury_cost_usd: float | None = None
    jury_duration_ms: int | None = None
    jury_degraded: bool | None = None
    # Debate calls that reported no cost (a known jury_cost_usd is then a
    # lower bound).
    unpriced_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _escalates(confidence: object, threshold: float) -> bool:
    """Jury routing: a non-finite confidence always escalates."""
    if not is_finite_number(confidence):
        return True
    return confidence < threshold  # type: ignore[operator]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[int], percent: float) -> int | None:
    """Nearest-rank percentile: an observed value, never an interpolation."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _check_unit_interval(name: str, value: object) -> float:
    if not is_finite_number(value) or not 0.0 <= value <= 1.0:  # type: ignore[operator]
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return float(value)  # type: ignore[arg-type]


def _check_cost(name: str, value: object) -> float:
    if not is_finite_number(value) or value < 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)  # type: ignore[arg-type]


def _debate_cost(
    verdict_cost: float | None, primary_cost: float | None
) -> float | None:
    """Cost the escalation added on top of the primary call.

    An escalated verdict's total includes the primary classifier's cost, so
    the debate part is the difference. Unknown when either side is unknown.
    """
    if verdict_cost is None or primary_cost is None:
        return None
    return verdict_cost - primary_cost


class EvaluationReport:
    """Measured primary and jury outcomes for a labelled dataset."""

    def __init__(self, items: list[EvaluationItem], band_upper: float) -> None:
        self.items = items
        self.band_upper = band_upper

    # -- summary ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Headline numbers for the evaluated dataset.

        Accuracy fields are None when their denominator is zero. Costs are in
        USD; ``debate_cost_usd`` sums the debates whose cost is known and is
        None when none is, and ``unpriced_calls`` counts debate calls that
        reported no cost (a non-zero count makes the sum a lower bound).
        Latencies are nearest-rank percentiles of the debated items'
        ``Jury.escalate`` duration.
        """
        items = self.items
        debated = [item for item in items if item.debated]
        known_costs = self._known_debate_costs()
        durations = [
            item.jury_duration_ms
            for item in debated
            if item.jury_duration_ms is not None
        ]
        fallbacks: dict[str, int] = {}
        for item in debated:
            strategy = item.jury_strategy or ""
            if "fallback" in strategy or strategy.startswith("cost_guard"):
                fallbacks[strategy] = fallbacks.get(strategy, 0) + 1

        return {
            "n": len(items),
            "band_upper": self.band_upper,
            "primary_accuracy": _ratio(
                sum(1 for item in items if item.primary_correct), len(items)
            ),
            "debated": len(debated),
            "jury_accuracy_on_debated": _ratio(
                sum(1 for item in debated if item.jury_correct), len(debated)
            ),
            "primary_accuracy_on_debated": _ratio(
                sum(1 for item in debated if item.primary_correct), len(debated)
            ),
            "flips_helped": sum(
                1 for item in debated if item.jury_correct and not item.primary_correct
            ),
            "flips_hurt": sum(
                1 for item in debated if item.primary_correct and not item.jury_correct
            ),
            "debate_cost_usd": sum(known_costs) if known_costs else None,
            "unpriced_calls": sum(item.unpriced_calls for item in debated),
            "mean_debate_cost_usd": (
                sum(known_costs) / len(known_costs) if known_costs else None
            ),
            "latency_ms_p50": _percentile(durations, 50),
            "latency_ms_p95": _percentile(durations, 95),
            "degraded": sum(1 for item in debated if item.jury_degraded),
            "fallbacks": {key: fallbacks[key] for key in sorted(fallbacks)},
            "confusion": self._confusion(),
        }

    def _confusion(self) -> dict[str, dict[str, dict[str, int]]]:
        """``{"primary"|"jury": {expected: {predicted: count}}}``.

        Both matrices cover every label seen in the data, sorted. The primary
        matrix counts all items; the jury matrix counts debated items only.
        """
        seen: set[str] = set()
        for item in self.items:
            seen.update((item.expected, item.primary_label))
            if item.debated and item.jury_label is not None:
                seen.add(item.jury_label)
        labels = sorted(seen)
        primary = {row: {col: 0 for col in labels} for row in labels}
        jury = {row: {col: 0 for col in labels} for row in labels}
        for item in self.items:
            primary[item.expected][item.primary_label] += 1
            if item.debated and item.jury_label is not None:
                jury[item.expected][item.jury_label] += 1
        return {"primary": primary, "jury": jury}

    def _known_debate_costs(self) -> list[float]:
        return [
            item.jury_cost_usd
            for item in self.items
            if item.debated and item.jury_cost_usd is not None
        ]

    # -- threshold sweep ----------------------------------------------------

    def _default_escalation_cost(self) -> float:
        """Measured mean debate cost, or 0.05 USD when no debate was priced."""
        known = self._known_debate_costs()
        if not known:
            return DEFAULT_ESCALATION_COST_USD
        return sum(known) / len(known)

    def _default_thresholds(self) -> list[float]:
        """The calibrator grid (0.5 to 0.95) up to ``band_upper``."""
        candidates = [t for t in DEFAULT_THRESHOLDS if t <= self.band_upper]
        return candidates or [self.band_upper]

    def threshold_sweep(
        self,
        thresholds: Sequence[float] | None = None,
        error_cost: float = 10.0,
        escalation_cost: float | None = None,
    ) -> list[dict[str, Any]]:
        """Replay the measured outcomes at each threshold.

        At threshold ``t`` an item escalates when its primary confidence is
        below ``t`` (or not finite) and then takes the jury's label; the
        others keep the primary label. ``total_cost = errors * error_cost +
        escalations * escalation_cost``. ``escalation_cost=None`` uses the
        measured mean debate cost, or 0.05 when no debate was priced.

        ``primary_accuracy`` is the primary's accuracy on the items it keeps
        and ``jury_accuracy`` the jury's on the items it gets; both are None
        when their group is empty.

        Raises ``ValueError`` for a threshold above ``band_upper``: items at
        or above ``band_upper`` were never debated, so their jury outcome is
        unknown.
        """
        candidates = (
            list(thresholds) if thresholds is not None else self._default_thresholds()
        )
        if not candidates:
            raise ValueError("at least one threshold is required")
        for threshold in candidates:
            _check_unit_interval("threshold", threshold)
            if threshold > self.band_upper:
                raise ValueError(
                    f"threshold {threshold} is above band_upper {self.band_upper}: "
                    "items at or above band_upper were not debated, so their jury "
                    "outcome was not measured. Re-run evaluate() with a higher "
                    "band_upper."
                )
        error_cost = _check_cost("error_cost", error_cost)
        per_escalation = (
            self._default_escalation_cost()
            if escalation_cost is None
            else _check_cost("escalation_cost", escalation_cost)
        )

        n = len(self.items)
        rows: list[dict[str, Any]] = []
        for threshold in candidates:
            escalations = errors = kept = kept_correct = jury_correct = 0
            for item in self.items:
                if _escalates(item.primary_confidence, threshold):
                    escalations += 1
                    if item.jury_correct:
                        jury_correct += 1
                    else:
                        errors += 1
                else:
                    kept += 1
                    if item.primary_correct:
                        kept_correct += 1
                    else:
                        errors += 1
            rows.append(
                {
                    "threshold": threshold,
                    "escalation_rate": _ratio(escalations, n),
                    "system_accuracy": _ratio(n - errors, n),
                    "jury_accuracy": _ratio(jury_correct, escalations),
                    "primary_accuracy": _ratio(kept_correct, kept),
                    "errors": errors,
                    "total_cost": errors * error_cost + escalations * per_escalation,
                }
            )
        return rows

    def best_threshold(
        self,
        error_cost: float = 10.0,
        escalation_cost: float | None = None,
        thresholds: Sequence[float] | None = None,
    ) -> float:
        """Threshold with the lowest sweep ``total_cost``; the first one wins ties."""
        rows = self.threshold_sweep(
            thresholds, error_cost=error_cost, escalation_cost=escalation_cost
        )
        best = rows[0]
        for row in rows[1:]:
            if row["total_cost"] < best["total_cost"]:
                best = row
        return float(best["threshold"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_upper": self.band_upper,
            "summary": self.summary(),
            "items": [item.to_dict() for item in self.items],
        }


async def _run_bounded(
    factories: list[Callable[[], Awaitable[T]]], concurrency: int
) -> list[T]:
    """Run coroutine factories with at most ``concurrency`` in flight.

    The first failure raises. No new call starts after it, and calls still
    queued or in flight are cancelled, so a failed evaluation stops spending.
    """
    semaphore = asyncio.Semaphore(concurrency)
    aborted = False

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        nonlocal aborted
        async with semaphore:
            if aborted:
                raise asyncio.CancelledError()
            try:
                return await factory()
            except BaseException:
                aborted = True
                raise

    tasks = [asyncio.ensure_future(run(factory)) for factory in factories]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        aborted = True
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class JuryEvaluator:
    """Runs a jury over labelled data and reports what it changed.

    The primary classifier is called exactly once per text. Items whose
    primary confidence is below ``band_upper`` (or not finite) go to
    ``jury.escalate``, bypassing the jury's own threshold and
    ``escalation_override``. ``jury.stats`` is not touched.
    """

    def __init__(self, jury: EvaluableJury) -> None:
        self.jury = jury

    async def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        band_upper: float = 0.95,
        max_escalations: int | None = None,
        concurrency: int = 5,
    ) -> EvaluationReport:
        """Classify every text, debate the ones below ``band_upper``, report.

        ``max_escalations`` caps the number of debates: when more items fall
        below ``band_upper``, ``TooManyEscalationsError`` (a ``ValueError``) is
        raised after the primary pass and before any debate starts. ``concurrency`` bounds the calls in
        flight in each pass.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        band_upper = _check_unit_interval("band_upper", band_upper)
        if isinstance(concurrency, bool) or not isinstance(concurrency, int):
            raise ValueError(
                f"concurrency must be an integer >= 1, got {concurrency!r}"
            )
        if concurrency < 1:
            raise ValueError(
                f"concurrency must be an integer >= 1, got {concurrency!r}"
            )
        if max_escalations is not None and (
            isinstance(max_escalations, bool)
            or not isinstance(max_escalations, int)
            or max_escalations < 0
        ):
            raise ValueError(
                f"max_escalations must be None or an integer >= 0, got {max_escalations!r}"
            )
        personas = getattr(self.jury, "personas", None)
        if personas is not None and not personas:
            raise ValueError(
                "JuryEvaluator needs a jury with at least one persona; this jury has none."
            )

        classifier = self.jury.classifier
        primaries = await _run_bounded(
            [partial(classifier.classify, text) for text in texts], concurrency
        )

        debated_indexes = [
            index
            for index, primary in enumerate(primaries)
            if _escalates(primary.confidence, band_upper)
        ]
        if max_escalations is not None and len(debated_indexes) > max_escalations:
            raise TooManyEscalationsError(
                f"{len(debated_indexes)} item(s) have a primary confidence below "
                f"band_upper={band_upper}, more than max_escalations={max_escalations}. "
                "No debate was run. Raise max_escalations or lower band_upper."
            )

        verdicts = await _run_bounded(
            [
                partial(self.jury.escalate, texts[index], primaries[index])
                for index in debated_indexes
            ],
            concurrency,
        )
        verdict_by_index = dict(zip(debated_indexes, verdicts, strict=True))

        items: list[EvaluationItem] = []
        for index, (text, expected, primary) in enumerate(
            zip(texts, labels, primaries, strict=True)
        ):
            item = EvaluationItem(
                text=text,
                expected=expected,
                primary_label=primary.label,
                primary_confidence=primary.confidence,
                primary_correct=primary.label == expected,
                primary_cost_usd=primary.cost_usd,
            )
            verdict = verdict_by_index.get(index)
            if verdict is not None:
                transcript = verdict.debate_transcript
                item.debated = True
                item.jury_label = verdict.label
                item.jury_confidence = verdict.confidence
                item.jury_correct = verdict.label == expected
                item.jury_strategy = verdict.judge_strategy
                item.jury_cost_usd = _debate_cost(
                    verdict.total_cost_usd, primary.cost_usd
                )
                item.jury_duration_ms = verdict.total_duration_ms
                item.jury_degraded = verdict.debate_degraded
                item.unpriced_calls = (
                    transcript.unpriced_calls if transcript is not None else 0
                )
            items.append(item)

        return EvaluationReport(items, band_upper)
