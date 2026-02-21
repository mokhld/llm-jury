from __future__ import annotations

import json
import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.personas.base import PersonaResponse
from tests.helpers import FakeLLMClient, FakeLLMReply


class LLMJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_judge_parses_json(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("unknown", 0.4),
            rounds=[[
                PersonaResponse("A", "unsafe", 0.9, "harmful", ["harm"]),
                PersonaResponse("B", "safe", 0.6, "context", ["context"]),
            ]],
            duration_ms=10,
            total_tokens=20,
            total_cost_usd=0.001,
        )
        client = FakeLLMClient(
            {
                "judge": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "unsafe",
                            "confidence": 0.81,
                            "reasoning": "Harm argument is stronger.",
                            "key_agreements": ["ambiguous text"],
                            "key_disagreements": ["intent"],
                            "decisive_factor": "targeted harm",
                        }
                    )
                )
            }
        )

        verdict = await LLMJudge(model="judge", llm_client=client).judge(
            transcript,
            ["safe", "unsafe"],
        )

        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 0.81)

    async def test_llm_judge_invalid_json_falls_back_to_primary(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("safe", 0.91),
            rounds=[[
                PersonaResponse("A", "unsafe", 0.9, "harmful", ["harm"]),
            ]],
            duration_ms=10,
            total_tokens=20,
            total_cost_usd=0.001,
        )
        client = FakeLLMClient({"judge": FakeLLMReply("not-json")})
        verdict = await LLMJudge(model="judge", llm_client=client).judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "safe")
        self.assertAlmostEqual(verdict.confidence, 0.91)


if __name__ == "__main__":
    unittest.main()
