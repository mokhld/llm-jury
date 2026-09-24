from __future__ import annotations

import json
import unittest

from llm_jury.classifiers.llm_classifier import LLMClassifier
from llm_jury.personas.schema import build_classifier_response_schema
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

    def test_llm_classifier_rejects_empty_labels(self) -> None:
        with self.assertRaises(ValueError):
            LLMClassifier(model="m", labels=[])

    def test_llm_classifier_rejects_whitespace_only_labels(self) -> None:
        with self.assertRaises(ValueError):
            LLMClassifier(model="m", labels=["", "  "])

    def test_llm_classifier_rejects_none_labels(self) -> None:
        with self.assertRaises(ValueError):
            LLMClassifier(model="m")


class LLMClassifierValidationTests(unittest.IsolatedAsyncioTestCase):
    """BUG-01/BUG-02: output is constrained and validated."""

    async def _classify(
        self, content: str, cost: float | None = 0.003
    ) -> tuple[object, FakeLLMClient]:
        client = FakeLLMClient({"classifier": FakeLLMReply(content, cost_usd=cost)})
        classifier = LLMClassifier(
            model="classifier", labels=["safe", "unsafe"], llm_client=client
        )
        return await classifier.classify("text"), client

    async def test_sends_strict_classifier_schema(self) -> None:
        _, client = await self._classify('{"label": "safe", "confidence": 0.9}')
        self.assertEqual(
            client.calls[0]["response_format"],
            build_classifier_response_schema(["safe", "unsafe"]),
        )

    async def test_label_is_canonicalised(self) -> None:
        result, _ = await self._classify('{"label": " Unsafe ", "confidence": 0.9}')
        self.assertEqual(result.label, "unsafe")
        self.assertAlmostEqual(result.confidence, 0.9)

    async def test_out_of_set_label_forces_escalation(self) -> None:
        raw = '{"label": "Unsafe - borderline", "confidence": 0.95}'
        result, _ = await self._classify(raw)
        self.assertEqual(result.label, "safe")  # labels[0]
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(
            result.raw_output, {"raw_content": raw, "error": "label_not_in_labels"}
        )
        self.assertAlmostEqual(result.cost_usd, 0.003)

    async def test_non_numeric_confidence_does_not_raise(self) -> None:
        raw = '{"label": "unsafe", "confidence": "low"}'
        result, _ = await self._classify(raw)
        self.assertEqual(result.label, "unsafe")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(
            result.raw_output, {"raw_content": raw, "error": "invalid_confidence"}
        )

    async def test_nan_confidence_forces_escalation(self) -> None:
        raw = '{"label": "Safe", "confidence": NaN}'
        result, _ = await self._classify(raw)
        self.assertEqual(result.label, "safe")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.raw_output["error"], "invalid_confidence")

    async def test_unreported_cost_stays_none(self) -> None:
        result, _ = await self._classify(
            '{"label": "safe", "confidence": 0.9}', cost=None
        )
        self.assertIsNone(result.cost_usd)


if __name__ == "__main__":
    unittest.main()
