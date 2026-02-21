from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from tests.helpers import FakeLLMClient, FakeLLMReply
from llm_jury.classifiers.base import ClassificationResult
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode, DebateTranscript
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona, PersonaResponse


def _make_persona(name: str = "TestPersona") -> Persona:
    return Persona(name=name, role="test", system_prompt="You are a test persona.", model="gpt-fake")


class ThresholdBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_confidence_equal_threshold_does_not_escalate(self) -> None:
        threshold = 0.7
        classifier = FunctionClassifier(lambda _: ("safe", threshold), ["safe", "unsafe"])
        jury = Jury(classifier=classifier, personas=[], confidence_threshold=threshold)

        verdict = await jury.classify("text")
        self.assertFalse(verdict.was_escalated)

    async def test_above_threshold_no_escalation(self) -> None:
        classifier = FunctionClassifier(lambda _: ("safe", 0.7001), ["safe", "unsafe"])
        jury = Jury(classifier=classifier, personas=[], confidence_threshold=0.7)

        verdict = await jury.classify("text")
        self.assertFalse(verdict.was_escalated)
        self.assertEqual(jury.stats.fast_path, 1)
        self.assertEqual(jury.stats.escalated, 0)

    async def test_below_threshold_escalates(self) -> None:
        persona_response = json.dumps(
            {
                "label": "unsafe",
                "confidence": 0.9,
                "reasoning": "Content is harmful.",
                "key_factors": ["harmful language"],
            }
        )
        client = FakeLLMClient(
            responses={"gpt-fake": FakeLLMReply(content=persona_response)}
        )
        personas = [_make_persona()]

        classifier = FunctionClassifier(lambda _: ("safe", 0.6999), ["safe", "unsafe"])
        jury = Jury(
            classifier=classifier,
            personas=personas,
            confidence_threshold=0.7,
            judge=MajorityVoteJudge(),
            llm_client=client,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        verdict = await jury.classify("some risky text")
        self.assertTrue(verdict.was_escalated)
        self.assertEqual(jury.stats.escalated, 1)
        self.assertEqual(jury.stats.fast_path, 0)


if __name__ == "__main__":
    unittest.main()
