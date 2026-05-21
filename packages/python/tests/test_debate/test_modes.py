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
            "A": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.8,
                        "reasoning": "a",
                        "key_factors": ["k1"],
                    }
                )
            ),
            "B": FakeLLMReply(
                json.dumps(
                    {
                        "label": "unsafe",
                        "confidence": 0.8,
                        "reasoning": "b",
                        "key_factors": ["k2"],
                    }
                )
            ),
            "C": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.8,
                        "reasoning": "c",
                        "key_factors": ["k3"],
                    }
                )
            ),
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
                "A": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "safe",
                            "confidence": 0.9,
                            "reasoning": "a",
                            "key_factors": ["x"],
                        }
                    )
                ),
                "B": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "safe",
                            "confidence": 0.8,
                            "reasoning": "b",
                            "key_factors": ["y"],
                        }
                    )
                ),
                "C": FakeLLMReply(
                    json.dumps(
                        {
                            "label": "safe",
                            "confidence": 0.85,
                            "reasoning": "c",
                            "key_factors": ["z"],
                        }
                    )
                ),
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

    async def test_one_persona_failure_does_not_crash_independent_mode(self) -> None:
        llm = _FlakyLLMClient(fail_for={"B"})
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.INDEPENDENT, max_rounds=1),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        self.assertEqual(len(transcript.rounds[0]), 3)
        names = [r.persona_name for r in transcript.rounds[0]]
        self.assertEqual(names, ["A", "B", "C"])
        failed = next(r for r in transcript.rounds[0] if r.persona_name == "B")
        self.assertEqual(failed.confidence, 0.0)
        self.assertIn("Persona call failed", failed.reasoning)
        # Non-failed personas must succeed — locks the response_format plumbing
        # in _FlakyLLMClient. Without it, every persona would silently fall back
        # and this assertion would catch the regression.
        for name in ("A", "C"):
            ok = next(r for r in transcript.rounds[0] if r.persona_name == name)
            self.assertAlmostEqual(ok.confidence, 0.8)
            self.assertEqual(ok.label, "safe")

    async def test_one_persona_failure_does_not_crash_sequential_mode(self) -> None:
        llm = _FlakyLLMClient(fail_for={"B"})
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.SEQUENTIAL),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        self.assertEqual(len(transcript.rounds[0]), 3)
        failed = next(r for r in transcript.rounds[0] if r.persona_name == "B")
        self.assertIn("Persona call failed", failed.reasoning)

    async def test_empty_personas_returns_zero_round_transcript(self) -> None:
        # T1: debate(personas=[]) should short-circuit at engine.py:91 without
        # calling the LLM. Guards against future regressions that try to index
        # self.personas before the empty-list check.
        llm = FakeLLMClient()
        engine = DebateEngine(
            personas=[],
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=3),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        self.assertEqual(transcript.rounds, [])
        self.assertEqual(transcript.total_tokens, 0)
        self.assertEqual(transcript.total_cost_usd, 0.0)
        self.assertIsNone(transcript.summary)
        self.assertEqual(llm.calls, [])

    async def test_f7_early_stops_on_high_confidence_even_with_split_labels(
        self,
    ) -> None:
        """F7: with early_stop_min_confidence set, the loop halts after a
        round in which every persona is highly confident in its own answer,
        even if they disagree on label."""
        # Round 1: split labels (safe vs unsafe), all high confidence.
        responses = {
            "A": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.97,
                        "reasoning": "a",
                        "key_factors": ["x"],
                    }
                )
            ),
            "B": FakeLLMReply(
                json.dumps(
                    {
                        "label": "unsafe",
                        "confidence": 0.96,
                        "reasoning": "b",
                        "key_factors": ["y"],
                    }
                )
            ),
            "C": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.98,
                        "reasoning": "c",
                        "key_factors": ["z"],
                    }
                )
            ),
        }
        llm = FakeLLMClient(responses)
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(
                mode=DebateMode.DELIBERATION,
                max_rounds=5,
                early_stop_min_confidence=0.95,
            ),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        # Round 0 (initial opinions) always runs; F7 short-circuits at
        # the consensus check after round 1 — so 2 rounds, not 5.
        self.assertEqual(len(transcript.rounds), 2)

    async def test_f7_does_not_early_stop_when_one_persona_below_threshold(
        self,
    ) -> None:
        """F7: MIN-confidence semantics — a single low-confidence persona
        keeps the loop going."""
        responses = {
            "A": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.97,
                        "reasoning": "a",
                        "key_factors": ["x"],
                    }
                )
            ),
            "B": FakeLLMReply(
                json.dumps(
                    {
                        "label": "unsafe",
                        "confidence": 0.50,  # below threshold
                        "reasoning": "b",
                        "key_factors": ["y"],
                    }
                )
            ),
            "C": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.96,
                        "reasoning": "c",
                        "key_factors": ["z"],
                    }
                )
            ),
        }
        llm = FakeLLMClient(responses)
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(
                mode=DebateMode.DELIBERATION,
                max_rounds=3,
                early_stop_min_confidence=0.95,
            ),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        # Threshold not met → loop runs to max_rounds.
        self.assertEqual(len(transcript.rounds), 3)

    async def test_f7_default_preserves_unanimous_only_consensus(self) -> None:
        """F7 is opt-in: without early_stop_min_confidence, split labels keep
        the loop going to max_rounds regardless of confidence."""
        responses = {
            "A": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.99,
                        "reasoning": "a",
                        "key_factors": [],
                    }
                )
            ),
            "B": FakeLLMReply(
                json.dumps(
                    {
                        "label": "unsafe",
                        "confidence": 0.99,
                        "reasoning": "b",
                        "key_factors": [],
                    }
                )
            ),
            "C": FakeLLMReply(
                json.dumps(
                    {
                        "label": "safe",
                        "confidence": 0.99,
                        "reasoning": "c",
                        "key_factors": [],
                    }
                )
            ),
        }
        llm = FakeLLMClient(responses)
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=3),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        # Default behaviour: no early-stop without unanimous labels → 3 rounds.
        self.assertEqual(len(transcript.rounds), 3)

    async def test_one_persona_failure_does_not_crash_deliberation(self) -> None:
        llm = _FlakyLLMClient(fail_for={"B"})
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=2),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        # First round still has all three personas (B as fallback)
        self.assertEqual(len(transcript.rounds[0]), 3)
        failed = next(r for r in transcript.rounds[0] if r.persona_name == "B")
        self.assertEqual(failed.confidence, 0.0)
        self.assertIn("Persona call failed", failed.reasoning)


class _FlakyLLMClient:
    """LLM client that raises only when a target persona makes the call.

    Personas in these tests have system_prompt equal to their name ("A", "B", "C"),
    so exact-match keeps the flaky behaviour scoped to persona calls and never
    fires on the summarisation prompt.

    Accepts ``response_format`` because real LLM clients (and LiteLLMClient)
    receive it from the debate engine; without it the call signature would
    mismatch and every persona would silently fall back instead of only the
    targeted ones.
    """

    def __init__(self, fail_for: set[str]) -> None:
        self.fail_for = fail_for

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.0,
        response_format=None,
    ):
        if system_prompt in self.fail_for:
            raise RuntimeError(
                f"simulated upstream failure for persona {system_prompt}"
            )
        return {
            "content": json.dumps(
                {
                    "label": "safe",
                    "confidence": 0.8,
                    "reasoning": "ok",
                    "key_factors": ["k"],
                }
            ),
            "tokens": 10,
            "cost_usd": 0.001,
        }


if __name__ == "__main__":
    unittest.main()
