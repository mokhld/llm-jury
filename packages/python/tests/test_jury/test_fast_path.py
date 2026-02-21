from __future__ import annotations

import unittest

from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.jury.core import Jury


class JuryFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_confidence_skips_escalation(self) -> None:
        classifier = FunctionClassifier(lambda text: ("safe", 0.95), ["safe", "unsafe"])
        jury = Jury(classifier=classifier, personas=[], confidence_threshold=0.7)

        verdict = await jury.classify("hello")

        self.assertFalse(verdict.was_escalated)
        self.assertEqual(verdict.label, "safe")
        self.assertIsNone(verdict.debate_transcript)
        self.assertEqual(jury.stats.total, 1)
        self.assertEqual(jury.stats.fast_path, 1)
        self.assertEqual(jury.stats.escalated, 0)


if __name__ == "__main__":
    unittest.main()
