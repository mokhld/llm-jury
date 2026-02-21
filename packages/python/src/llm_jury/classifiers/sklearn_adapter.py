from __future__ import annotations

from typing import Any

from .base import ClassificationResult, Classifier


class SklearnClassifier(Classifier):
    def __init__(self, model: Any, labels: list[str], vectorizer: Any = None) -> None:
        self.model = model
        self.labels = labels
        self.vectorizer = vectorizer

    async def classify(self, text: str) -> ClassificationResult:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy is required for SklearnClassifier") from exc

        features = self.vectorizer.transform([text]) if self.vectorizer else [text]
        probs = self.model.predict_proba(features)[0]
        idx = int(np.argmax(probs))
        return ClassificationResult(
            label=self.labels[idx],
            confidence=float(probs[idx]),
            raw_output=list(probs),
        )
