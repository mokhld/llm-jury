from __future__ import annotations

import json
import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateConfig, DebateEngine, DebateMode
from llm_jury.personas.base import Persona
from llm_jury.personas.schema import build_persona_response_schema
from tests.helpers import FakeLLMClient, FakeLLMReply


class StructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    """F2: persona calls forward a JSON Schema response_format."""

    def setUp(self) -> None:
        # Unique system prompts so FakeLLMClient's substring matcher routes
        # cleanly — short single-letter tokens collide with words like
        # "Available" inside the prompt template.
        self.personas = [
            Persona(name="Alpha", role="role", system_prompt="ALPHA_PROMPT"),
            Persona(name="Bravo", role="role", system_prompt="BRAVO_PROMPT"),
        ]
        self.primary = ClassificationResult(label="unknown", confidence=0.3)
        self.labels = ["safe", "unsafe"]

    async def test_persona_call_forwards_response_format(self) -> None:
        llm = FakeLLMClient(
            {
                "ALPHA_PROMPT": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "safe",
                            "confidence": 0.9,
                            "reasoning": "a",
                            "key_factors": [],
                            "dissent_notes": None,
                        }
                    )
                ),
                "BRAVO_PROMPT": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "safe",
                            "confidence": 0.8,
                            "reasoning": "b",
                            "key_factors": [],
                            "dissent_notes": None,
                        }
                    )
                ),
            }
        )
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.INDEPENDENT, max_rounds=1),
        )

        await engine.debate("text", self.primary, self.labels)

        expected = build_persona_response_schema(self.labels)
        self.assertGreaterEqual(len(llm.calls), 2)
        for call in llm.calls:
            self.assertEqual(call["response_format"], expected)

    async def test_null_dissent_notes_becomes_none(self) -> None:
        llm = FakeLLMClient(
            {
                "ALPHA_PROMPT": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "safe",
                            "confidence": 0.9,
                            "reasoning": "a",
                            "key_factors": [],
                            "dissent_notes": None,
                        }
                    )
                ),
                "BRAVO_PROMPT": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "safe",
                            "confidence": 0.9,
                            "reasoning": "b",
                            "key_factors": [],
                            "dissent_notes": "actual rebuttal",
                        }
                    )
                ),
            }
        )
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.INDEPENDENT, max_rounds=1),
        )

        transcript = await engine.debate("text", self.primary, self.labels)
        round_responses = {r.persona_name: r for r in transcript.rounds[0]}

        self.assertIsNone(round_responses["Alpha"].dissent_notes)
        self.assertEqual(round_responses["Bravo"].dissent_notes, "actual rebuttal")


if __name__ == "__main__":
    unittest.main()
