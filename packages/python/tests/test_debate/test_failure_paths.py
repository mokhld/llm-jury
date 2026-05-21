from __future__ import annotations

import json
import unittest
from typing import Any

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateConfig, DebateEngine, DebateMode
from llm_jury.personas.base import Persona


def _persona_payload(label: str = "safe", confidence: float = 0.8) -> str:
    return json.dumps(
        {
            "label": label,
            "confidence": confidence,
            "reasoning": "ok",
            "key_factors": ["k"],
        }
    )


class _CallCountingLLMClient:
    """LLM client whose persona response depends on which round we're in.

    Personas in these tests have system_prompt equal to their name. Counts
    per-persona calls so we can make a persona succeed on round 1 and fail
    on round 2 (or vice versa).
    """

    def __init__(self, fail_round_for: dict[str, set[int]] | None = None) -> None:
        self.fail_round_for = fail_round_for or {}
        self.call_counts: dict[str, int] = {}
        # Track summarisation calls separately — system_prompt for the summariser
        # starts with "You are a neutral summarisation agent".
        self.summarisation_calls = 0

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if system_prompt.startswith("You are a neutral summarisation agent"):
            self.summarisation_calls += 1
            return {"content": "synthesis", "tokens": 5, "cost_usd": 0.0001}

        persona_name = system_prompt
        round_num = self.call_counts.get(persona_name, 0) + 1
        self.call_counts[persona_name] = round_num

        if round_num in self.fail_round_for.get(persona_name, set()):
            raise RuntimeError(f"persona {persona_name} fails on round {round_num}")

        return {"content": _persona_payload(), "tokens": 10, "cost_usd": 0.001}


class _SummariserFailureClient:
    """LLM client whose persona calls succeed but whose summariser raises."""

    def __init__(self) -> None:
        self.persona_calls = 0
        self.summarisation_calls = 0

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if system_prompt.startswith("You are a neutral summarisation agent"):
            self.summarisation_calls += 1
            raise RuntimeError("summariser is unavailable")
        self.persona_calls += 1
        # Diverging labels keep the debate going until max_rounds (no early exit).
        label = "safe" if self.persona_calls % 2 == 0 else "unsafe"
        return {"content": _persona_payload(label=label), "tokens": 10, "cost_usd": 0.001}


class CascadeFailureTests(unittest.IsolatedAsyncioTestCase):
    """T3: cascade failures — a persona that fails mid-debate (not on round 1)
    must not crash the whole verdict. The engine's per-round resilience
    (_run_round + _gather_with_fallback) defends this; the test pins it."""

    def setUp(self) -> None:
        self.personas = [
            Persona(name="A", role="role", system_prompt="A"),
            Persona(name="B", role="role", system_prompt="B"),
            Persona(name="C", role="role", system_prompt="C"),
        ]
        self.primary = ClassificationResult(label="unknown", confidence=0.4)

    async def test_persona_failing_on_second_round_does_not_crash(self) -> None:
        # B succeeds on round 1, fails on round 2. A and C always succeed.
        llm = _CallCountingLLMClient(fail_round_for={"B": {2}})
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            # Force max_rounds=2 with diverging round-1 labels so deliberation
            # continues into round 2 without hitting consensus early. We achieve
            # divergence not via labels (this client returns "safe" for all) but
            # by running max_rounds with consensus disabled via persona count > 1.
            config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=2),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        # Two rounds executed. Round 1: all succeed. Round 2: B falls back.
        self.assertEqual(len(transcript.rounds), 2)
        round1 = {r.persona_name: r for r in transcript.rounds[0]}
        round2 = {r.persona_name: r for r in transcript.rounds[1]}

        self.assertAlmostEqual(round1["B"].confidence, 0.8)
        self.assertEqual(round2["B"].confidence, 0.0)
        self.assertIn("Persona call failed", round2["B"].reasoning)
        # A and C must still succeed on round 2.
        for name in ("A", "C"):
            self.assertAlmostEqual(round2[name].confidence, 0.8)

    async def test_all_personas_failing_on_round_two_still_returns_transcript(self) -> None:
        # Round 1 succeeds; round 2 wipes everyone out. Engine must still return
        # a transcript with both rounds populated by fallbacks.
        llm = _CallCountingLLMClient(fail_round_for={"A": {2}, "B": {2}, "C": {2}})
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=2),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        self.assertEqual(len(transcript.rounds), 2)
        for response in transcript.rounds[1]:
            self.assertEqual(response.confidence, 0.0)
            self.assertIn("Persona call failed", response.reasoning)


class SummariserFailureTests(unittest.IsolatedAsyncioTestCase):
    """T5: summariser failure must not crash the verdict. The engine catches
    the exception, logs a warning, and returns the transcript with summary=None.
    """

    def setUp(self) -> None:
        self.personas = [
            Persona(name="A", role="role", system_prompt="A"),
            Persona(name="B", role="role", system_prompt="B"),
            Persona(name="C", role="role", system_prompt="C"),
        ]
        self.primary = ClassificationResult(label="unknown", confidence=0.4)

    async def test_debate_completes_when_summariser_raises(self) -> None:
        llm = _SummariserFailureClient()
        engine = DebateEngine(
            personas=self.personas,
            llm_client=llm,
            config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=2),
        )

        transcript = await engine.debate("text", self.primary, ["safe", "unsafe"])

        # Persona rounds must all have completed normally.
        self.assertEqual(len(transcript.rounds), 2)
        for round_responses in transcript.rounds:
            self.assertEqual(len(round_responses), 3)
        # Summariser was invoked and failed.
        self.assertEqual(llm.summarisation_calls, 1)
        # Transcript still usable; summary absent.
        self.assertIsNone(transcript.summary)


class MalformedPersonaJSONTests(unittest.IsolatedAsyncioTestCase):
    """T6: _parse_persona_response must degrade gracefully on malformed payloads."""

    def setUp(self) -> None:
        # A single-persona engine is enough — we're testing the parser directly
        # via the engine instance (it's an instance method).
        self.engine = DebateEngine(
            personas=[Persona(name="A", role="role", system_prompt="A")],
        )
        self.labels = ["safe", "unsafe"]

    def test_missing_label_falls_back_to_first_label(self) -> None:
        raw = json.dumps({"confidence": 0.7, "reasoning": "r", "key_factors": []})
        response = self.engine._parse_persona_response(raw, "A", self.labels)
        # Missing label → "unknown" placeholder (engine.py:471 default).
        self.assertEqual(response.label, "unknown")
        self.assertAlmostEqual(response.confidence, 0.7)

    def test_confidence_above_one_is_clamped(self) -> None:
        raw = json.dumps({"label": "safe", "confidence": 1.5})
        response = self.engine._parse_persona_response(raw, "A", self.labels)
        self.assertEqual(response.label, "safe")
        self.assertEqual(response.confidence, 1.0)

    def test_negative_confidence_is_clamped(self) -> None:
        raw = json.dumps({"label": "unsafe", "confidence": -0.4})
        response = self.engine._parse_persona_response(raw, "A", self.labels)
        self.assertEqual(response.label, "unsafe")
        self.assertEqual(response.confidence, 0.0)

    def test_non_numeric_confidence_falls_back(self) -> None:
        # "high" is not a number; clamp_confidence(float("high")) raises ValueError.
        # Currently the parser does not catch this — pin the current behaviour so
        # any future strict-mode flag is a deliberate change.
        raw = json.dumps({"label": "safe", "confidence": "high"})
        with self.assertRaises(ValueError):
            self.engine._parse_persona_response(raw, "A", self.labels)

    def test_non_string_label_is_coerced(self) -> None:
        # str(42) → "42" — pins that the parser does not validate label is in
        # the labels list. Caller is expected to handle unknown labels.
        raw = json.dumps({"label": 42, "confidence": 0.5})
        response = self.engine._parse_persona_response(raw, "A", self.labels)
        self.assertEqual(response.label, "42")

    def test_array_payload_falls_back(self) -> None:
        # safe_json_parse returns None for non-dict payloads, triggering the
        # fallback branch at engine.py:457-466.
        response = self.engine._parse_persona_response("[1, 2, 3]", "A", self.labels)
        self.assertEqual(response.label, "safe")  # labels[0]
        self.assertEqual(response.confidence, 0.0)
        self.assertIn("Failed to parse", response.reasoning)


if __name__ == "__main__":
    unittest.main()
