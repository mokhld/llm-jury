from __future__ import annotations

import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.weighted_vote import WeightedVoteJudge
from llm_jury.personas.base import PersonaResponse


class WeightedVoteJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_weighted_vote_prefers_higher_confidence(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("unknown", 0.2),
            rounds=[[
                PersonaResponse("A", "safe", 0.99, "r1", ["a"]),
                PersonaResponse("B", "unsafe", 0.6, "r2", ["b"]),
                PersonaResponse("C", "unsafe", 0.2, "r3", ["c"]),
            ]],
            duration_ms=10,
            total_tokens=20,
            total_cost_usd=0.001,
        )

        verdict = await WeightedVoteJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "safe")
        self.assertGreater(verdict.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
