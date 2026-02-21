from __future__ import annotations

import json
import unittest

from llm_jury.classifiers.llm_classifier import LLMClassifier
from tests.helpers import FakeLLMClient, FakeLLMReply


class LLMClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_classifier_parses_json(self) -> None:
        client = FakeLLMClient(
            {
                "classifier": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "unsafe",
                            "confidence": 0.83,
                        }
                    )
                )
            }
        )
        classifier = LLMClassifier(
            model="classifier",
            labels=["safe", "unsafe"],
            llm_client=client,
        )

        result = await classifier.classify("text")
        self.assertEqual(result.label, "unsafe")
        self.assertAlmostEqual(result.confidence, 0.83)

    async def test_llm_classifier_invalid_json_falls_back(self) -> None:
        client = FakeLLMClient({"classifier": FakeLLMReply("not-json")})
        classifier = LLMClassifier(
            model="classifier",
            labels=["safe", "unsafe"],
            llm_client=client,
        )

        result = await classifier.classify("text")
        self.assertEqual(result.label, "safe")
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
