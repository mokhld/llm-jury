from __future__ import annotations

import asyncio
from typing import Any

from .base import ClassificationResult, Classifier


def _column_labels(model: Any, labels: list[str]) -> list[str]:
    """Label for each ``predict_proba`` column.

    scikit-learn orders the columns by ``model.classes_``. When the model exposes
    ``classes_`` and its entries are the same set of strings as ``labels``, the
    columns are named by class. Otherwise ``labels`` names the columns by
    position, which lets ``labels`` give display names to non-string classes.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return list(labels)
    class_names = [str(c) for c in classes]
    if len(class_names) != len(labels):
        raise ValueError(
            f"model.classes_ has {len(class_names)} entries ({class_names}) but "
            f"{len(labels)} labels were given ({labels}). Pass one label per "
            "predict_proba column, in model.classes_ order."
        )
    if set(class_names) == set(labels) and len(set(class_names)) == len(class_names):
        return class_names
    return list(labels)


class SklearnClassifier(Classifier):
    """scikit-learn style model with ``predict_proba`` as the primary classifier.

    Inference runs in a worker thread so it does not block the event loop, and
    results report ``cost_usd=0.0`` because no paid API is called.
    """

    def __init__(self, model: Any, labels: list[str], vectorizer: Any = None) -> None:
        self.model = model
        self.labels = labels
        self.vectorizer = vectorizer
        self._column_labels = _column_labels(model, labels)

    async def classify(self, text: str) -> ClassificationResult:
        return await asyncio.to_thread(self._classify_sync, text)

    def _classify_sync(self, text: str) -> ClassificationResult:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy is required for SklearnClassifier") from exc

        features = self.vectorizer.transform([text]) if self.vectorizer else [text]
        probs = self.model.predict_proba(features)[0]
        idx = int(np.argmax(probs))
        return ClassificationResult(
            label=self._column_labels[idx],
            confidence=float(probs[idx]),
            raw_output=[float(p) for p in probs],
            cost_usd=0.0,
        )
