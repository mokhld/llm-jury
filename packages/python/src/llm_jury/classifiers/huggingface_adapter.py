from __future__ import annotations

import asyncio
from typing import Any

from .base import ClassificationResult, Classifier


def _score_list(output: Any) -> list[dict]:
    """The ``[{"label", "score"}, ...]`` list for one input.

    With ``top_k`` set at construction, the pipeline wraps a single input's
    scores in an outer list; a call-time ``top_k`` returns them unwrapped.
    """
    if isinstance(output, dict):
        return [output]
    if isinstance(output, list) and output and isinstance(output[0], list):
        return output[0]
    return list(output or [])


class HuggingFaceClassifier(Classifier):
    """Local ``transformers`` text-classification pipeline as the primary classifier.

    ``labels`` sets the label list the jury debates over. When it is omitted, the
    label names come from the model's full score list on the first call.
    Inference runs in a worker thread so it does not block the event loop, and
    results report ``cost_usd=0.0`` because no paid API is called.
    """

    def __init__(
        self, model_name: str, device: str = "cpu", labels: list[str] | None = None
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for HuggingFaceClassifier"
            ) from exc

        self.pipe = pipeline(
            "text-classification", model=model_name, device=device, top_k=None
        )
        self.labels = list(labels) if labels else []

    async def classify(self, text: str) -> ClassificationResult:
        output = await asyncio.to_thread(self.pipe, text)
        results = _score_list(output)
        if not results:
            raise RuntimeError("HuggingFace pipeline returned no scores")
        top = max(results, key=lambda x: x["score"])

        if not self.labels and len(results) > 1:
            self.labels = [item["label"] for item in results]

        return ClassificationResult(
            label=top["label"],
            confidence=float(top["score"]),
            raw_output=results,
            cost_usd=0.0,
        )
