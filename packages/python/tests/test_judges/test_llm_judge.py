from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.base import Verdict
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.personas.base import PersonaResponse
from llm_jury.personas.schema import build_judge_response_schema
from tests.helpers import FakeLLMClient, FakeLLMReply


class LLMJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_judge_parses_json(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("unknown", 0.4),
            rounds=[
                [
                    PersonaResponse("A", "unsafe", 0.9, "harmful", ["harm"]),
                    PersonaResponse("B", "safe", 0.6, "context", ["context"]),
                ]
            ],
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

    async def test_llm_judge_invalid_json_falls_back_to_vote(self) -> None:
        transcript = DebateTranscript(
            input_text="text",
            primary_result=ClassificationResult("safe", 0.91),
            rounds=[
                [
                    PersonaResponse("A", "unsafe", 0.9, "harmful", ["harm"]),
                ]
            ],
            duration_ms=10,
            total_tokens=20,
            total_cost_usd=0.001,
        )
        client = FakeLLMClient({"judge": FakeLLMReply("not-json")})
        verdict = await LLMJudge(model="judge", llm_client=client).judge(
            transcript, ["safe", "unsafe"]
        )
        # The debate's vote wins over the primary result it escalated from.
        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 1.0)
        self.assertEqual(verdict.judge_strategy, "llm_judge_fallback_invalid_json")
        self.assertTrue(
            verdict.reasoning.startswith("LLM judge response was not valid JSON.")
        )


LABELS = ["safe", "unsafe"]


def _judge_reply(**overrides: Any) -> str:
    data: dict[str, Any] = {
        "label": "unsafe",
        "confidence": 0.8,
        "reasoning": "Harm argument is stronger.",
        "key_agreements": ["ambiguous text"],
        "key_disagreements": ["intent"],
        "decisive_factor": "targeted harm",
    }
    data.update(overrides)
    return json.dumps(data)


def _debate(
    rounds: list[list[PersonaResponse]] | None = None,
    total_cost_usd: float | None = 0.01,
    persona_biases: dict[str, str] | None = None,
) -> DebateTranscript:
    return DebateTranscript(
        input_text="text",
        primary_result=ClassificationResult("safe", 0.4),
        rounds=rounds
        or [
            [
                PersonaResponse("A", "unsafe", 0.9, "harmful", ["harm"]),
                PersonaResponse("B", "unsafe", 0.7, "risky", ["risk"]),
                PersonaResponse("C", "safe", 0.6, "context", ["context"]),
            ]
        ],
        duration_ms=10,
        total_tokens=20,
        total_cost_usd=total_cost_usd,
        persona_biases=persona_biases or {},
    )


class _RaisingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        raise RuntimeError("400 Bad Request: context length exceeded")


class LLMJudgeValidationTests(unittest.IsolatedAsyncioTestCase):
    async def _judge(
        self, content: str, cost: float | None = 0.002, **kwargs: Any
    ) -> tuple[Verdict, FakeLLMClient]:
        client = FakeLLMClient({"judge": FakeLLMReply(content, cost_usd=cost)})
        verdict = await LLMJudge(model="judge", llm_client=client).judge(
            _debate(**kwargs), LABELS
        )
        return verdict, client

    async def test_sends_strict_judge_schema(self) -> None:
        _, client = await self._judge(_judge_reply())
        self.assertEqual(
            client.calls[0]["response_format"], build_judge_response_schema(LABELS)
        )

    async def test_label_is_canonicalised(self) -> None:
        verdict, _ = await self._judge(_judge_reply(label="UNSAFE"))
        self.assertEqual(verdict.label, "unsafe")
        self.assertEqual(verdict.judge_strategy, "llm_judge")

    async def test_out_of_set_label_falls_back_to_vote(self) -> None:
        # BUG-01 reproduction: the judge answered 'Unsafe - borderline'.
        verdict, _ = await self._judge(_judge_reply(label="Unsafe - borderline"))
        self.assertEqual(verdict.judge_strategy, "llm_judge_fallback_invalid_label")
        self.assertEqual(verdict.label, "unsafe")  # 2 of 3 final-round votes
        self.assertAlmostEqual(verdict.confidence, 2 / 3)
        self.assertTrue(
            verdict.reasoning.startswith(
                "LLM judge returned label 'Unsafe - borderline', which is not one "
                "of the configured labels."
            )
        )
        self.assertIsNone(verdict.judge_details)

    async def test_nan_confidence_falls_back_to_vote(self) -> None:
        content = _judge_reply().replace('"confidence": 0.8', '"confidence": NaN')
        verdict, _ = await self._judge(content)
        self.assertEqual(
            verdict.judge_strategy, "llm_judge_fallback_invalid_confidence"
        )
        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 2 / 3)

    async def test_non_numeric_confidence_falls_back_to_vote(self) -> None:
        verdict, _ = await self._judge(_judge_reply(confidence="very high"))
        self.assertEqual(
            verdict.judge_strategy, "llm_judge_fallback_invalid_confidence"
        )

    async def test_vote_fallback_uses_final_round_only(self) -> None:
        rounds = [
            [
                PersonaResponse("A", "unsafe", 0.9, "r1", []),
                PersonaResponse("B", "unsafe", 0.9, "r1", []),
            ],
            [
                PersonaResponse("A", "safe", 0.9, "r2", []),
                PersonaResponse("B", "safe", 0.8, "r2", []),
            ],
        ]
        verdict, _ = await self._judge("not json", rounds=rounds)
        self.assertEqual(verdict.label, "safe")
        self.assertAlmostEqual(verdict.confidence, 1.0)

    async def test_vote_fallback_without_valid_final_round_returns_primary(
        self,
    ) -> None:
        failed = PersonaResponse(
            "B", "safe", 0.0, "Persona call failed", [], failed=True
        )
        rounds = [
            [PersonaResponse("A", "unsafe", 0.9, "r1", [])],
            [failed],
        ]
        verdict, _ = await self._judge("not json", rounds=rounds)
        self.assertEqual(verdict.judge_strategy, "llm_judge_fallback_invalid_json")
        self.assertEqual(verdict.label, "safe")
        self.assertAlmostEqual(verdict.confidence, 0.4)
        self.assertTrue(
            verdict.reasoning.startswith("LLM judge response was not valid JSON.")
        )

    async def test_judge_details_are_recorded(self) -> None:
        verdict, _ = await self._judge(_judge_reply())
        self.assertEqual(
            verdict.judge_details,
            {
                "key_agreements": ["ambiguous text"],
                "key_disagreements": ["intent"],
                "decisive_factor": "targeted harm",
            },
        )
        self.assertEqual(verdict.to_dict()["judge_details"], verdict.judge_details)
        self.assertEqual(
            json.loads(verdict.to_json())["judge_details"]["decisive_factor"],
            "targeted harm",
        )

    async def test_total_cost_adds_debate_and_judge(self) -> None:
        verdict, _ = await self._judge(_judge_reply(), cost=0.002, total_cost_usd=0.01)
        self.assertAlmostEqual(verdict.total_cost_usd, 0.012)

    async def test_total_cost_unknown_when_nothing_priced(self) -> None:
        verdict, _ = await self._judge(_judge_reply(), cost=None, total_cost_usd=None)
        self.assertIsNone(verdict.total_cost_usd)

    async def test_judge_cost_counts_when_debate_cost_unknown(self) -> None:
        verdict, _ = await self._judge(_judge_reply(), cost=0.002, total_cost_usd=None)
        self.assertAlmostEqual(verdict.total_cost_usd, 0.002)

    async def test_prompt_lists_expert_roster(self) -> None:
        _, client = await self._judge(
            _judge_reply(),
            persona_biases={"A": "policy-strict", "C": "tends permissive"},
        )
        prompt = client.calls[0]["prompt"]
        self.assertIn("Expert roster:", prompt)
        self.assertIn("- A (known bias: policy-strict)", prompt)
        self.assertIn("- C (known bias: tends permissive)", prompt)

    async def test_prompt_omits_roster_without_biases(self) -> None:
        _, client = await self._judge(_judge_reply())
        self.assertNotIn("Expert roster", client.calls[0]["prompt"])


class LLMJudgeErrorFallbackTests(unittest.IsolatedAsyncioTestCase):
    """BUG-03: a judge LLM failure must not throw away the paid debate."""

    async def test_judge_exception_falls_back_to_vote(self) -> None:
        client = _RaisingClient()
        with self.assertLogs("llm_jury.judges.llm_judge", level="WARNING"):
            verdict = await LLMJudge(llm_client=client).judge(
                _debate(total_cost_usd=0.07), LABELS
            )

        self.assertEqual(client.calls, 1)
        self.assertEqual(verdict.judge_strategy, "llm_judge_fallback_error")
        self.assertEqual(verdict.label, "unsafe")
        self.assertAlmostEqual(verdict.confidence, 2 / 3)
        self.assertTrue(verdict.reasoning.startswith("LLM judge call failed"))
        self.assertIn("RuntimeError", verdict.reasoning)
        # The debate's spend is still reported.
        self.assertAlmostEqual(verdict.total_cost_usd, 0.07)

    async def test_cancellation_is_not_swallowed(self) -> None:
        class _CancelledClient:
            async def complete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await LLMJudge(llm_client=_CancelledClient()).judge(_debate(), LABELS)


if __name__ == "__main__":
    unittest.main()
