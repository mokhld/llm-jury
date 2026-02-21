"""Legal compliance classification with sequential debate and weighted vote.

Requires OPENAI_API_KEY in your environment.
"""
from __future__ import annotations

import asyncio

from llm_jury import Jury, PersonaRegistry, DebateConfig, DebateMode
from llm_jury.classifiers import LLMClassifier
from llm_jury.judges import WeightedVoteJudge


async def main() -> None:
    classifier = LLMClassifier(
        labels=["compliant", "non_compliant", "needs_review"],
    )

    jury = Jury(
        classifier=classifier,
        personas=PersonaRegistry.legal_compliance(),
        confidence_threshold=0.8,
        judge=WeightedVoteJudge(),
        debate_config=DebateConfig(mode=DebateMode.SEQUENTIAL),
    )

    verdict = await jury.classify(
        "Guaranteed double-digit returns with no downside risk."
    )

    print(f"Label:      {verdict.label}")
    print(f"Confidence: {verdict.confidence:.2f}")
    print(f"Escalated:  {verdict.was_escalated}")
    print(f"Strategy:   {verdict.judge_strategy}")
    print(f"Reasoning:  {verdict.reasoning[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
