from __future__ import annotations

from .base import ClassificationResult, Classifier


class HuggingFaceClassifier(Classifier):
    def __init__(self, model_name: str, device: str = "cpu", labels: list[str] | None = None) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for HuggingFaceClassifier") from exc

        self.pipe = pipeline("text-classification", model=model_name, device=device, top_k=None)
        self.labels = labels or []

    async def classify(self, text: str) -> ClassificationResult:
        results = self.pipe(text)[0]
        top = max(results, key=lambda x: x["score"])

        if not self.labels:
            self.labels = [item["label"] for item in results]

        return ClassificationResult(
            label=top["label"],
            confidence=float(top["score"]),
            raw_output=results,
        )
