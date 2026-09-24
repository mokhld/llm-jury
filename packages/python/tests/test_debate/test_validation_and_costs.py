from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from typing import Any

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import (
    DebateConfig,
    DebateEngine,
    DebateMode,
    exceeds_cost_cap,
    guard_spend,
)
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.personas.base import Persona

_SUMMARISER_PREFIX = "You are a neutral summarisation agent"
LABELS = ["safe", "unsafe"]


def _payload(label: Any = "safe", confidence: Any = 0.8) -> str:
    return json.dumps(
        {
            "label": label,
            "confidence": confidence,
            "reasoning": f"reasoning for {label}",
            "key_factors": ["k"],
        }
    )


class _ScriptedClient:
    """Answers each call from ``script(persona_name, round_number)``.

    Personas use their name as system prompt. The summariser gets
    ``summary_reply`` (or raises when it is an exception).
    """

    def __init__(
        self,
        script: Callable[[str, int], dict[str, Any]],
        summary_reply: dict[str, Any] | Exception | None = None,
    ) -> None:
        self.script = script
        self.summary_reply = summary_reply or {
            "content": "synthesis",
            "tokens": 5,
            "cost_usd": 0.0001,
        }
        self.persona_calls: dict[str, int] = {}
        self.summariser_calls = 0
        self.prompts: list[str] = []

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.prompts.append(prompt)
        if system_prompt.startswith(_SUMMARISER_PREFIX):
            self.summariser_calls += 1
            if isinstance(self.summary_reply, Exception):
                raise self.summary_reply
            return dict(self.summary_reply)
        round_number = self.persona_calls.get(system_prompt, 0) + 1
        self.persona_calls[system_prompt] = round_number
        return self.script(system_prompt, round_number)

    @property
    def total_persona_calls(self) -> int:
        return sum(self.persona_calls.values())


def _personas(*names: str, biases: dict[str, str] | None = None) -> list[Persona]:
    biases = biases or {}
    return [
        Persona(name=n, role="role", system_prompt=n, known_bias=biases.get(n))
        for n in names
    ]


PRIMARY = ClassificationResult(label="safe", confidence=0.4)


class PersonaLabelValidationTests(unittest.IsolatedAsyncioTestCase):
    """BUG-01: persona labels must be one of the configured labels."""

    async def test_case_variant_label_is_canonicalised(self) -> None:
        llm = _ScriptedClient(
            lambda name, _r: {"content": _payload("Unsafe", 0.9), "cost_usd": 0.001}
        )
        engine = DebateEngine(
            _personas("A"), DebateConfig(mode=DebateMode.INDEPENDENT), llm
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        response = transcript.rounds[0][0]
        self.assertFalse(response.failed)
        self.assertEqual(response.label, "unsafe")

    async def test_out_of_set_label_is_a_failed_response(self) -> None:
        raw = _payload("Unsafe - borderline", 0.9)
        llm = _ScriptedClient(lambda name, _r: {"content": raw, "cost_usd": 0.001})
        engine = DebateEngine(
            _personas("A"), DebateConfig(mode=DebateMode.INDEPENDENT), llm
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        response = transcript.rounds[0][0]
        self.assertTrue(response.failed)
        self.assertEqual(
            response.reasoning,
            "Persona returned label 'Unsafe - borderline', which is not one of "
            "the configured labels.",
        )
        self.assertEqual(response.raw_response, raw)
        # The paid call still counts towards cost.
        self.assertAlmostEqual(response.cost_usd, 0.001)
        self.assertEqual(transcript.persona_failures, 1)

    async def test_out_of_set_votes_never_reach_the_verdict(self) -> None:
        def script(name: str, _r: int) -> dict[str, Any]:
            label = "Unsafe - borderline" if name in ("A", "B") else "safe"
            return {"content": _payload(label, 0.9), "cost_usd": 0.001}

        engine = DebateEngine(
            _personas("A", "B", "C"),
            DebateConfig(mode=DebateMode.INDEPENDENT),
            _ScriptedClient(script),
        )
        transcript = await engine.debate("text", PRIMARY, LABELS)

        verdict = await MajorityVoteJudge().judge(transcript, LABELS)
        self.assertEqual(verdict.label, "safe")
        self.assertIn(verdict.label, LABELS)


class PersonaConfidenceValidationTests(unittest.IsolatedAsyncioTestCase):
    """BUG-02: a non-finite or non-numeric persona confidence is not a vote."""

    def setUp(self) -> None:
        self.engine = DebateEngine(_personas("A"))

    def test_nan_literal_confidence_is_failed(self) -> None:
        raw = '{"label": "unsafe", "confidence": NaN, "reasoning": "r"}'
        response = self.engine._parse_persona_response(raw, "A", LABELS)
        self.assertTrue(response.failed)
        self.assertEqual(response.confidence, 0.0)

    def test_missing_confidence_is_failed(self) -> None:
        raw = json.dumps({"label": "unsafe", "reasoning": "r"})
        response = self.engine._parse_persona_response(raw, "A", LABELS)
        self.assertTrue(response.failed)

    def test_numeric_string_confidence_is_accepted(self) -> None:
        raw = json.dumps({"label": "unsafe", "confidence": "0.65"})
        response = self.engine._parse_persona_response(raw, "A", LABELS)
        self.assertFalse(response.failed)
        self.assertAlmostEqual(response.confidence, 0.65)


class DebateCostAccountingTests(unittest.IsolatedAsyncioTestCase):
    """BUG-04: unknown costs stay unknown and are counted, not coerced to 0."""

    async def test_unpriced_client_reports_none_and_counts_calls(self) -> None:
        llm = _ScriptedClient(
            lambda name, _r: {"content": _payload("safe"), "cost_usd": None},
        )
        engine = DebateEngine(
            _personas("A", "B"), DebateConfig(mode=DebateMode.INDEPENDENT), llm
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertIsNone(transcript.total_cost_usd)
        self.assertEqual(transcript.unpriced_calls, 2)
        for response in transcript.rounds[0]:
            self.assertIsNone(response.cost_usd)

    async def test_mixed_costs_sum_known_and_count_unknown(self) -> None:
        def script(name: str, _r: int) -> dict[str, Any]:
            cost = None if name == "B" else 0.002
            return {"content": _payload("safe"), "cost_usd": cost}

        engine = DebateEngine(
            _personas("A", "B", "C"),
            DebateConfig(mode=DebateMode.INDEPENDENT),
            _ScriptedClient(script),
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertAlmostEqual(transcript.total_cost_usd, 0.004)
        self.assertEqual(transcript.unpriced_calls, 1)

    async def test_summariser_cost_is_included(self) -> None:
        def script(name: str, _r: int) -> dict[str, Any]:
            label = "unsafe" if name == "A" else "safe"
            return {"content": _payload(label), "cost_usd": 0.001}

        llm = _ScriptedClient(
            script, summary_reply={"content": "s", "tokens": 1, "cost_usd": 0.5}
        )
        engine = DebateEngine(
            _personas("A", "B"),
            DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=2),
            llm,
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertEqual(llm.summariser_calls, 1)
        # 2 personas x 2 rounds x 0.001 + 0.5 summariser
        self.assertAlmostEqual(transcript.total_cost_usd, 0.504)
        self.assertEqual(transcript.unpriced_calls, 0)

    async def test_failed_summariser_counts_as_unpriced_call(self) -> None:
        def script(name: str, _r: int) -> dict[str, Any]:
            label = "unsafe" if name == "A" else "safe"
            return {"content": _payload(label), "cost_usd": 0.001}

        llm = _ScriptedClient(script, summary_reply=RuntimeError("down"))
        engine = DebateEngine(
            _personas("A", "B"),
            DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=2),
            llm,
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertIsNone(transcript.summary)
        self.assertAlmostEqual(transcript.total_cost_usd, 0.004)
        self.assertEqual(transcript.unpriced_calls, 1)


class MidFlightCostGuardTests(unittest.IsolatedAsyncioTestCase):
    """BUG-04: the mid-flight guard prices unpriced calls at the estimate."""

    async def test_sequential_guard_trips_on_unpriced_calls(self) -> None:
        llm = _ScriptedClient(
            lambda name, _r: {"content": _payload("safe"), "cost_usd": None},
        )
        engine = DebateEngine(
            _personas("A", "B", "C", "D"),
            DebateConfig(mode=DebateMode.SEQUENTIAL),
            llm,
        )

        transcript = await engine.debate(
            "text",
            PRIMARY,
            LABELS,
            max_cost_usd=0.015,
            estimated_cost_per_call_usd=0.01,
        )

        # Spend after two calls is 0.02 > 0.015, so C and D never run.
        self.assertEqual(llm.total_persona_calls, 2)
        self.assertEqual(len(transcript.rounds[0]), 2)
        self.assertEqual(transcript.unpriced_calls, 2)

    async def test_deliberation_guard_skips_rounds_and_summariser(self) -> None:
        def script(name: str, _r: int) -> dict[str, Any]:
            label = "unsafe" if name == "A" else "safe"
            return {"content": _payload(label), "cost_usd": None}

        llm = _ScriptedClient(script)
        engine = DebateEngine(
            _personas("A", "B"),
            DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=3),
            llm,
        )

        transcript = await engine.debate(
            "text",
            PRIMARY,
            LABELS,
            max_cost_usd=0.01,
            estimated_cost_per_call_usd=0.01,
        )

        self.assertEqual(len(transcript.rounds), 1)
        self.assertEqual(llm.summariser_calls, 0)

    async def test_spend_exactly_at_cap_does_not_trip(self) -> None:
        # 0.1 + 0.2 is 0.30000000000000004 in floating point.
        costs = {"A": 0.1, "B": 0.2, "C": 0.0}
        llm = _ScriptedClient(
            lambda name, _r: {"content": _payload("safe"), "cost_usd": costs[name]},
        )
        engine = DebateEngine(
            _personas("A", "B", "C"),
            DebateConfig(mode=DebateMode.SEQUENTIAL),
            llm,
        )

        await engine.debate("text", PRIMARY, LABELS, max_cost_usd=0.3)

        self.assertEqual(llm.total_persona_calls, 3)

    def test_guard_helpers(self) -> None:
        self.assertAlmostEqual(guard_spend(None, 3, 0.01), 0.03)
        self.assertAlmostEqual(guard_spend(0.5, 0, 0.01), 0.5)
        self.assertAlmostEqual(guard_spend(0.5, 2, None), 0.5)
        self.assertFalse(exceeds_cost_cap(0.1 + 0.2, 0.3))
        self.assertTrue(exceeds_cost_cap(0.31, 0.3))
        self.assertFalse(exceeds_cost_cap(100.0, None))


class OpeningRoundConsensusTests(unittest.IsolatedAsyncioTestCase):
    """BUG-06: consensus is checked after the opening round too."""

    async def test_unanimous_opening_round_skips_deliberation_and_summary(
        self,
    ) -> None:
        llm = _ScriptedClient(
            lambda name, _r: {"content": _payload("unsafe"), "cost_usd": 0.001}
        )
        engine = DebateEngine(_personas("A", "B", "C"), llm_client=llm)  # defaults

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertEqual(len(transcript.rounds), 1)
        self.assertIsNone(transcript.summary)
        self.assertEqual(llm.total_persona_calls, 3)
        self.assertEqual(llm.summariser_calls, 0)

    async def test_consensus_in_a_later_round_still_summarises(self) -> None:
        # Split opening round, unanimous second round, max_rounds=4.
        def script(name: str, round_number: int) -> dict[str, Any]:
            label = "unsafe" if name == "A" and round_number == 1 else "safe"
            return {"content": _payload(label), "cost_usd": 0.001}

        llm = _ScriptedClient(script)
        engine = DebateEngine(
            _personas("A", "B"),
            DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=4),
            llm,
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertEqual(len(transcript.rounds), 2)
        self.assertEqual(transcript.summary, "synthesis")

    async def test_failed_personas_do_not_block_opening_consensus(self) -> None:
        def script(name: str, _r: int) -> dict[str, Any]:
            if name == "C":
                return {"content": "not json", "cost_usd": 0.001}
            return {"content": _payload("unsafe"), "cost_usd": 0.001}

        llm = _ScriptedClient(script)
        engine = DebateEngine(_personas("A", "B", "C"), llm_client=llm)

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertEqual(len(transcript.rounds), 1)


class PersonaBiasTests(unittest.IsolatedAsyncioTestCase):
    """BUG-10: the transcript records each persona's known bias."""

    async def test_persona_biases_are_recorded(self) -> None:
        llm = _ScriptedClient(
            lambda name, _r: {"content": _payload("safe"), "cost_usd": 0.001}
        )
        engine = DebateEngine(
            _personas("A", "B", biases={"A": "policy-strict"}),
            DebateConfig(mode=DebateMode.INDEPENDENT),
            llm,
        )

        transcript = await engine.debate("text", PRIMARY, LABELS)

        self.assertEqual(transcript.persona_biases, {"A": "policy-strict"})
        verdict = await MajorityVoteJudge().judge(transcript, LABELS)
        data = verdict.to_dict()["debate_transcript"]
        self.assertEqual(data["persona_biases"], {"A": "policy-strict"})
        self.assertEqual(data["unpriced_calls"], 0)


if __name__ == "__main__":
    unittest.main()
