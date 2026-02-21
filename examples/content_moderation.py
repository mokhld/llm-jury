"""Content moderation with LLM classifier and persona debate.

Requires OPENAI_API_KEY in your environment.
"""
from __future__ import annotations

import asyncio

from llm_jury import Jury, PersonaRegistry
from llm_jury.classifiers import LLMClassifier
from llm_jury.judges import MajorityVoteJudge


async def main() -> None:
    classifier = LLMClassifier(labels=["safe", "unsafe"])

    jury = Jury(
        classifier=classifier,
        personas=PersonaRegistry.content_moderation(),
        confidence_threshold=0.85,
        judge=MajorityVoteJudge(),
    )

    verdict = await jury.classify("Those people always ruin everything.")

    print(f"Label:      {verdict.label}")
    print(f"Confidence: {verdict.confidence:.2f}")
    print(f"Escalated:  {verdict.was_escalated}")
    print(f"Strategy:   {verdict.judge_strategy}")
    print(f"Reasoning:  {verdict.reasoning[:200]}")

    if verdict.debate_transcript:
        print(f"\nDebate ({len(verdict.debate_transcript.rounds)} round(s)):")
        for resp in verdict.debate_transcript.rounds[-1]:
            print(f"  {resp.persona_name}: {resp.label} ({resp.confidence:.2f})")
            print(f"    {resp.reasoning[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
