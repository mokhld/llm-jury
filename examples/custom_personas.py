"""Custom persona definitions with deliberation mode.

Requires OPENAI_API_KEY in your environment.
"""
from __future__ import annotations

import asyncio

from llm_jury import Jury, Persona, DebateConfig, DebateMode
from llm_jury.classifiers import LLMClassifier
from llm_jury.judges import MajorityVoteJudge


async def main() -> None:
    personas = [
        Persona(
            name="Strict Classifier",
            role="Applies policy literally",
            system_prompt="Prioritize hard policy matches over contextual uncertainty.",
        ),
        Persona(
            name="Context Advocate",
            role="Looks for benign interpretations",
            system_prompt="Focus on intent, quotation, and contextual nuance.",
        ),
        Persona(
            name="Harm Sentinel",
            role="Prioritizes minimizing harmful outcomes",
            system_prompt="Focus on downstream harm even under uncertainty.",
        ),
    ]

    classifier = LLMClassifier(labels=["allow", "review", "reject"])

    jury = Jury(
        classifier=classifier,
        personas=personas,
        confidence_threshold=0.8,
        judge=MajorityVoteJudge(),
        debate_config=DebateConfig(mode=DebateMode.DELIBERATION, max_rounds=2),
    )

    verdict = await jury.classify("That user is trash and should be removed.")

    print(f"Label:      {verdict.label}")
    print(f"Confidence: {verdict.confidence:.2f}")
    print(f"Escalated:  {verdict.was_escalated}")
    print(f"Strategy:   {verdict.judge_strategy}")

    if verdict.debate_transcript:
        print(f"\nRounds: {len(verdict.debate_transcript.rounds)}")
        for i, rnd in enumerate(verdict.debate_transcript.rounds):
            print(f"  Round {i + 1}:")
            for resp in rnd:
                print(f"    {resp.persona_name}: {resp.label} ({resp.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
