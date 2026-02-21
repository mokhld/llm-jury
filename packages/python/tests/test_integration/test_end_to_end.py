from __future__ import annotations

import json
import unittest

from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.registry import PersonaRegistry
from tests.helpers import FakeLLMClient, FakeLLMReply


class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_with_mock_llm(self) -> None:
        classifier = FunctionClassifier(lambda text: ("unknown", 0.45), ["safe", "unsafe"])
        responses = {
            "Policy Analyst": FakeLLMReply(
                json.dumps(
                    {
                        "label": "unsafe",
                        "confidence": 0.85,
                        "reasoning": "Violates policy",
                        "key_factors": ["explicit content"],
                    }
                )
            ),
            "Cultural Context Expert": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.6,
                        "reasoning": "Could be satire",
                        "key_factors": ["context"],
                    }
                )
            ),
            "Harm Assessment Specialist": FakeLLMReply(
                json.dumps(
                    {
                        "label": "unsafe",
                        "confidence": 0.75,
                        "reasoning": "Potential harm to vulnerable group",
                        "key_factors": ["vulnerable group"],
                    }
                )
            ),
            "judge": FakeLLMReply(
                json.dumps(
                    {
                        "label": "unsafe",
                        "confidence": 0.8,
                        "reasoning": "Majority agrees and vulnerable group risk dominates.",
                        "key_agreements": ["ambiguous text"],
                        "key_disagreements": ["satire context"],
                        "decisive_factor": "vulnerable group targeting",
                    }
                )
            ),
        }
        llm = FakeLLMClient(responses)

        jury = Jury(
            classifier=classifier,
            personas=PersonaRegistry.content_moderation(),
            confidence_threshold=0.7,
            judge=LLMJudge(model="judge", llm_client=llm),
            llm_client=llm,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        verdict = await jury.classify("some borderline content")

        self.assertTrue(verdict.was_escalated)
        self.assertEqual(verdict.label, "unsafe")
        self.assertGreater(verdict.confidence, 0.7)
        self.assertIn("vulnerable group", verdict.reasoning)
        self.assertEqual(len(verdict.debate_transcript.rounds), 1)
        self.assertEqual(len(verdict.debate_transcript.rounds[0]), 3)


if __name__ == "__main__":
    unittest.main()
