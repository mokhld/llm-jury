from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ClassificationResult:
    label: str
    confidence: float
    raw_output: Any = None
    cost_usd: float | None = None


class Classifier(ABC):
    labels: list[str]

    @abstractmethod
    async def classify(self, text: str) -> ClassificationResult:
        raise NotImplementedError

    async def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        return [await self.classify(text) for text in texts]
