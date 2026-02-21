from __future__ import annotations

import unittest

from llm_jury.classifiers.function_adapter import FunctionClassifier


class FunctionClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_classification(self) -> None:
        classifier = FunctionClassifier(
            fn=lambda text: ("safe", 0.9),
            labels=["safe", "unsafe"],
        )

        result = await classifier.classify("hello")

        self.assertEqual(result.label, "safe")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.raw_output, {"label": "safe", "confidence": 0.9})

    async def test_batch_classification(self) -> None:
        classifier = FunctionClassifier(
            fn=lambda text: ("unsafe", 0.7 if "!" in text else 0.8),
            labels=["safe", "unsafe"],
        )

        results = await classifier.classify_batch(["a", "b!"])

        self.assertEqual([r.label for r in results], ["unsafe", "unsafe"])
        self.assertEqual([r.confidence for r in results], [0.8, 0.7])


if __name__ == "__main__":
    unittest.main()
