from __future__ import annotations

import asyncio
import json
import unittest

from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona
from tests.helpers import FakeLLMClient, FakeLLMReply


def _make_persona(name: str = "TestPersona") -> Persona:
    return Persona(
        name=name,
        role="test",
        system_prompt="You are a test persona.",
        model="gpt-fake",
    )


class JuryBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_classification(self) -> None:
        classifier = FunctionClassifier(
            lambda text: ("unsafe", 0.2) if "!" in text else ("safe", 0.95),
            ["safe", "unsafe"],
        )
        jury = Jury(classifier=classifier, personas=[], confidence_threshold=0.7)

        verdicts = await jury.classify_batch(["hello", "bad!"], concurrency=2)

        self.assertEqual([v.label for v in verdicts], ["safe", "unsafe"])
        self.assertEqual(jury.stats.total, 2)

    async def test_batch_preserves_input_order(self) -> None:
        async def classify_fn(text: str):
            if text == "slow":
                await asyncio.sleep(0.02)
            elif text == "fast":
                await asyncio.sleep(0.001)
            return (text, 0.99)

        classifier = FunctionClassifier(classify_fn, ["slow", "fast"])
        jury = Jury(classifier=classifier, personas=[], confidence_threshold=0.7)

        verdicts = await jury.classify_batch(["slow", "fast"], concurrency=2)
        self.assertEqual([v.label for v in verdicts], ["slow", "fast"])

    async def test_batch_mixed_fast_path_and_escalation(self) -> None:
        persona_response = json.dumps(
            {
                "label": "unsafe",
                "confidence": 0.9,
                "reasoning": "Flagged as harmful.",
                "key_factors": ["harmful content"],
            }
        )
        client = FakeLLMClient(
            responses={"gpt-fake": FakeLLMReply(content=persona_response)}
        )

        classifier = FunctionClassifier(
            lambda text: ("safe", 0.95) if text == "hello" else ("unsafe", 0.3),
            ["safe", "unsafe"],
        )
        jury = Jury(
            classifier=classifier,
            personas=[_make_persona()],
            confidence_threshold=0.7,
            judge=MajorityVoteJudge(),
            llm_client=client,
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        )

        verdicts = await jury.classify_batch(["hello", "bad stuff"], concurrency=2)

        self.assertEqual(len(verdicts), 2)
        self.assertFalse(verdicts[0].was_escalated)
        self.assertTrue(verdicts[1].was_escalated)
        self.assertEqual(jury.stats.fast_path, 1)
        self.assertEqual(jury.stats.escalated, 1)
        self.assertEqual(jury.stats.total, 2)


class FailFastBatchTests(unittest.IsolatedAsyncioTestCase):
    """BUG-07: once a fail-fast batch rejects, it stops spending."""

    def _jury(self, started: list[str], finished: list[str]) -> Jury:
        async def classify_fn(text: str):
            started.append(text)
            if text == "boom":
                raise RuntimeError("classifier exploded")
            await asyncio.sleep(0.05)
            finished.append(text)
            return ("safe", 0.95)

        classifier = FunctionClassifier(classify_fn, ["safe", "unsafe"])
        return Jury(classifier=classifier, personas=[], confidence_threshold=0.7)

    async def test_no_new_classify_starts_after_a_failure(self) -> None:
        started: list[str] = []
        finished: list[str] = []
        jury = self._jury(started, finished)

        with self.assertRaises(RuntimeError):
            await jury.classify_batch(["boom", "a", "b", "c"], concurrency=1)
        # Give any leaked task time to run.
        await asyncio.sleep(0.1)

        self.assertEqual(started, ["boom"])

    async def test_in_flight_calls_are_cancelled(self) -> None:
        started: list[str] = []
        finished: list[str] = []
        jury = self._jury(started, finished)

        with self.assertRaises(RuntimeError):
            await jury.classify_batch(["slow", "boom", "a", "b", "c"], concurrency=2)
        await asyncio.sleep(0.1)

        self.assertEqual(started, ["slow", "boom"])
        self.assertEqual(finished, [])

    async def test_return_exceptions_still_runs_every_text(self) -> None:
        started: list[str] = []
        finished: list[str] = []
        jury = self._jury(started, finished)

        results = await jury.classify_batch(
            ["boom", "a", "b"], concurrency=1, return_exceptions=True
        )

        self.assertIsInstance(results[0], RuntimeError)
        self.assertEqual(sorted(finished), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
