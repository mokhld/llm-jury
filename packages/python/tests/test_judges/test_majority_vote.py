from __future__ import annotations

import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.personas.base import PersonaResponse


class MajorityVoteJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_unanimous(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("unknown", 0.4),
            rounds=[[
                PersonaResponse("A", "unsafe", 0.9, "r1", ["a"]),
                PersonaResponse("B", "unsafe", 0.8, "r2", ["b"]),
                PersonaResponse("C", "unsafe", 0.7, "r3", ["c"]),
            ]],
            duration_ms=10,
            total_tokens=20,
            total_cost_usd=0.001,
        )

        verdict = await MajorityVoteJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")
        self.assertEqual(verdict.confidence, 1.0)

    async def test_split(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("unknown", 0.4),
            rounds=[[
                PersonaResponse("A", "unsafe", 0.9, "r1", ["a"]),
                PersonaResponse("B", "safe", 0.8, "r2", ["b"]),
                PersonaResponse("C", "unsafe", 0.7, "r3", ["c"]),
            ]],
            duration_ms=10,
            total_tokens=20,
            total_cost_usd=0.001,
        )

        verdict = await MajorityVoteJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 2 / 3, places=3)


if __name__ == "__main__":
    unittest.main()
