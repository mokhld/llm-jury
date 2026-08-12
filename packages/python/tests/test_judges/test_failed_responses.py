from __future__ import annotations

import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.bayesian import BayesianJudge
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.judges.weighted_vote import WeightedVoteJudge
from llm_jury.personas.base import PersonaResponse
from tests.helpers import FakeLLMClient


def _failed(persona_name: str, label: str = "safe") -> PersonaResponse:
    return PersonaResponse(
        persona_name=persona_name,
        label=label,
        confidence=0.0,
        reasoning="Persona call failed: AuthenticationError: no api key",
        key_factors=[],
        failed=True,
    )


def _transcript(rounds: list[list[PersonaResponse]]) -> DebateTranscript:
    return DebateTranscript(
        input_text="text",
        primary_result=ClassificationResult("unsafe", 0.55),
        rounds=rounds,
        duration_ms=10,
        total_tokens=20,
        total_cost_usd=0.001,
    )


class FailedResponsesAreNotVotesTests(unittest.IsolatedAsyncioTestCase):
    """Failed persona placeholders must not count as votes in any judge.

    Regression guard: a debate where every persona call failed (e.g. missing
    API key) used to yield a unanimous labels[0] verdict at confidence 1.0
    under majority vote, and the OPPOSITE label at ~1.0 under Bayesian.
    """

    async def test_majority_all_failed_falls_back_to_primary(self) -> None:
        transcript = _transcript([[_failed("A"), _failed("B"), _failed("C")]])
        verdict = await MajorityVoteJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 0.55)
        self.assertIn("failed", verdict.reasoning)

    async def test_majority_ignores_failed_votes(self) -> None:
        transcript = _transcript(
            [
                [
                    PersonaResponse("A", "unsafe", 0.9, "r1", ["a"]),
                    PersonaResponse("B", "unsafe", 0.8, "r2", ["b"]),
                    _failed("C"),
                ]
            ]
        )
        verdict = await MajorityVoteJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")
        # Confidence is computed over the two real votes, not three.
        self.assertAlmostEqual(verdict.confidence, 1.0)

    async def test_majority_failed_votes_cannot_flip_the_winner(self) -> None:
        # One real "unsafe" vote vs two failed "safe" placeholders: the real
        # vote must win even though the placeholders share labels[0].
        transcript = _transcript(
            [
                [
                    PersonaResponse("A", "unsafe", 0.9, "r1", ["a"]),
                    _failed("B"),
                    _failed("C"),
                ]
            ]
        )
        verdict = await MajorityVoteJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")

    async def test_weighted_all_failed_falls_back_to_primary(self) -> None:
        transcript = _transcript([[_failed("A"), _failed("B")]])
        verdict = await WeightedVoteJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 0.55)

    async def test_bayesian_all_failed_falls_back_to_primary(self) -> None:
        # Regression: failed responses (confidence 0.0) used to push the
        # posterior away from labels[0], electing "unsafe"... or rather the
        # opposite of the placeholder label, at high confidence.
        transcript = _transcript([[_failed("A"), _failed("B"), _failed("C")]])
        verdict = await BayesianJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 0.55)
        self.assertIn("failed", verdict.reasoning)

    async def test_bayesian_ignores_failed_votes(self) -> None:
        transcript = _transcript(
            [
                [
                    PersonaResponse("A", "safe", 0.9, "r1", ["a"]),
                    _failed("B"),
                ]
            ]
        )
        verdict = await BayesianJudge().judge(transcript, ["safe", "unsafe"])
        self.assertEqual(verdict.label, "safe")

    async def test_llm_judge_skips_call_when_all_failed(self) -> None:
        transcript = _transcript([[_failed("A"), _failed("B")]])
        client = FakeLLMClient()
        judge = LLMJudge(llm_client=client)

        verdict = await judge.judge(transcript, ["safe", "unsafe"])

        self.assertEqual(verdict.label, "unsafe")
        self.assertEqual(verdict.judge_strategy, "llm_judge_fallback_personas_failed")
        # No LLM call was made — nothing real to judge.
        self.assertEqual(client.calls, [])

    async def test_llm_judge_prompt_excludes_failed_responses(self) -> None:
        transcript = _transcript(
            [
                [
                    PersonaResponse("A", "unsafe", 0.9, "real reasoning", ["a"]),
                    _failed("B"),
                ]
            ]
        )
        judge = LLMJudge(llm_client=FakeLLMClient())
        prompt = judge._build_prompt(transcript, ["safe", "unsafe"])
        self.assertIn("real reasoning", prompt)
        self.assertNotIn("Persona call failed", prompt)


if __name__ == "__main__":
    unittest.main()
