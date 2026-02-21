from __future__ import annotations

import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.bayesian import BayesianJudge
from llm_jury.personas.base import PersonaResponse


class BayesianJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_bayesian_falls_back_without_responses(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("safe", 0.77),
            rounds=[[]],
            duration_ms=10,
            total_tokens=0,
            total_cost_usd=0.0,
        )

        verdict = await BayesianJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "safe")
        self.assertEqual(verdict.confidence, 0.77)

    async def test_bayesian_aggregates_posteriors(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("safe", 0.4),
            rounds=[[
                PersonaResponse("A", "unsafe", 0.9, "harm", ["h"]),
                PersonaResponse("B", "unsafe", 0.8, "risk", ["r"]),
                PersonaResponse("C", "safe", 0.55, "context", ["c"]),
            ]],
            duration_ms=10,
            total_tokens=0,
            total_cost_usd=0.0,
        )

        priors = {
            "A": {"unsafe": 0.8, "safe": 0.2},
            "B": {"unsafe": 0.7, "safe": 0.3},
            "C": {"unsafe": 0.3, "safe": 0.7},
        }
        verdict = await BayesianJudge(priors).judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")
        self.assertGreater(verdict.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
