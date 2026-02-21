from __future__ import annotations

import inspect
from typing import Awaitable, Callable

from .base import ClassificationResult, Classifier


class FunctionClassifier(Classifier):
    def __init__(
        self,
        fn: Callable[[str], tuple[str, float]] | Callable[[str], Awaitable[tuple[str, float]]],
        labels: list[str],
    ) -> None:
        self.fn = fn
        self.labels = labels

    async def classify(self, text: str) -> ClassificationResult:
        result = self.fn(text)
        if inspect.isawaitable(result):
            label, confidence = await result
        else:
            label, confidence = result

        return ClassificationResult(
            label=label,
            confidence=float(confidence),
            raw_output={"label": label, "confidence": float(confidence)},
        )
