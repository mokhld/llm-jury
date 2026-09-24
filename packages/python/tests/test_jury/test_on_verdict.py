from __future__ import annotations

import json
import unittest
from typing import Any

from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.judges.base import Verdict
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona
from tests.helpers import FakeLLMClient, FakeLLMReply

LABELS = ["safe", "unsafe"]


def _persona_reply(cost: float) -> FakeLLMReply:
    return FakeLLMReply(
        json.dumps(
            {"label": "unsafe", "confidence": 0.9, "reasoning": "r", "key_factors": []}
        ),
        cost_usd=cost,
    )


class OnVerdictTests(unittest.IsolatedAsyncioTestCase):
    """BUG-05: on_verdict fires exactly once for every verdict classify returns."""

    async def _run(self, confidence: float, **kwargs: Any) -> tuple[Verdict, list]:
        seen: list[Verdict] = []
        kwargs.setdefault(
            "llm_client", FakeLLMClient({"persona": _persona_reply(0.001)})
        )
        jury = Jury(
            classifier=FunctionClassifier(lambda _: ("safe", confidence), LABELS),
            personas=[
                Persona(name="A", role="r", system_prompt="persona-A", model="persona")
            ],
            judge=MajorityVoteJudge(),
            debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
            on_verdict=seen.append,
            **kwargs,
        )
        verdict = await jury.classify("text")
        return verdict, seen

    def _assert_fired_once(self, seen: list[Verdict], verdict: Verdict) -> None:
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], verdict)

    async def test_fast_path(self) -> None:
        verdict, seen = await self._run(0.95)
        self.assertEqual(verdict.judge_strategy, "primary_classifier")
        self._assert_fired_once(seen, verdict)

    async def test_user_override(self) -> None:
        verdict, seen = await self._run(0.4, on_cost_estimate=lambda _e, _t: False)
        self.assertEqual(verdict.judge_strategy, "cost_guard_user_override")
        self._assert_fired_once(seen, verdict)

    async def test_pre_flight(self) -> None:
        verdict, seen = await self._run(0.4, max_debate_cost_usd=0.0001)
        self.assertEqual(verdict.judge_strategy, "cost_guard_pre_flight")
        self._assert_fired_once(seen, verdict)

    async def test_primary_fallback(self) -> None:
        verdict, seen = await self._run(
            0.4,
            max_debate_cost_usd=0.5,
            llm_client=FakeLLMClient({"persona": _persona_reply(0.9)}),
        )
        self.assertEqual(verdict.judge_strategy, "cost_guard_primary_fallback")
        self._assert_fired_once(seen, verdict)

    async def test_judged(self) -> None:
        verdict, seen = await self._run(0.4)
        self.assertEqual(verdict.judge_strategy, "majority_vote")
        self._assert_fired_once(seen, verdict)


if __name__ == "__main__":
    unittest.main()
