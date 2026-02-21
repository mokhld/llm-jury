from __future__ import annotations

import json
import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateConfig, DebateEngine, DebateMode
from llm_jury.personas.base import Persona
from tests.helpers import FakeLLMClient, FakeLLMReply


class DebateModeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.personas = [
            Persona(name="A", role="role", system_prompt="A"),
            Persona(name="B", role="role", system_prompt="B"),
            Persona(name="C", role="role", system_prompt="C"),
        ]
        self.primary = ClassificationResult(label="unknown", confidence=0.4)

    async def test_independent_mode_runs_single_round(self) -> None:
        responses = {
            "A": FakeLLMReply(json.dumps({"label": "safe", "confidence": 0.8, "reasoning": "a", "key_factors": ["k1"]})),
            "B": FakeLLMReply(json.dumps({"label": "unsafe", "confidence": 0.8, "reasoning": "b", "key_factors": ["k2"]})),
            "C": FakeLLMReply(json.dumps({"label": "safe", "confidence": 0.8, "reasoning": "c", "key_factors": ["k3"]})),
        }
        llm = FakeLLMClient(responses)
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.INDEPENDENT, max_rounds=1),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        self.assertEqual(len(transcript.rounds), 1)
        self.assertEqual(len(transcript.rounds[0]), 3)

    async def test_sequential_mode_runs_single_round(self) -> None:
        llm = FakeLLMClient()
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.SEQUENTIAL),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        self.assertEqual(len(transcript.rounds), 1)
        self.assertEqual(len(transcript.rounds[0]), 3)

    async def test_deliberation_stops_on_consensus(self) -> None:
        llm = FakeLLMClient(
            {
                "A": FakeLLMReply(json.dumps({"label": "safe", "confidence": 0.9, "reasoning": "a", "key_factors": ["x"]})),
                "B": FakeLLMReply(json.dumps({"label": "safe", "confidence": 0.8, "reasoning": "b", "key_factors": ["y"]})),
                "C": FakeLLMReply(json.dumps({"label": "safe", "confidence": 0.85, "reasoning": "c", "key_factors": ["z"]})),
            }
        )
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=3),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        self.assertEqual(len(transcript.rounds), 2)
        self.assertIsNotNone(transcript.summary)


if __name__ == "__main__":
    unittest.main()
