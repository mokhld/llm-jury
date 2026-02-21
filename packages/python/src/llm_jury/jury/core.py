from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

from llm_jury.classifiers.base import ClassificationResult, Classifier
from llm_jury.debate.engine import DebateConfig, DebateEngine
from llm_jury.judges.base import JudgeStrategy, Verdict
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.llm.client import LLMClient, LiteLLMClient
from llm_jury.personas.base import Persona


@dataclass(slots=True)
class JuryStats:
    total: int = 0
    fast_path: int = 0
    escalated: int = 0

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.total if self.total > 0 else 0.0

    @property
    def cost_savings_vs_always_escalate(self) -> float:
        return self.fast_path / self.total if self.total > 0 else 0.0


class Jury:
    def __init__(
        self,
        classifier: Classifier,
        personas: list[Persona],
        confidence_threshold: float = 0.7,
        judge: JudgeStrategy | None = None,
        debate_config: DebateConfig | None = None,
        escalation_override: Callable[[ClassificationResult], bool] | None = None,
        max_debate_cost_usd: float | None = None,
        on_escalation: Callable[[str, ClassificationResult], None] | None = None,
        on_verdict: Callable[[Verdict], None] | None = None,
        logger: logging.Logger | None = None,
        llm_client: LLMClient | None = None,
        debate_concurrency: int = 5,
    ) -> None:
        self.classifier = classifier
        self.personas = personas
        self.threshold = confidence_threshold
        self._llm_client = llm_client or LiteLLMClient()
        self.judge = judge or LLMJudge(llm_client=self._llm_client)
        self.debate_config = debate_config or DebateConfig()
        self.debate_engine = DebateEngine(
            personas,
            self.debate_config,
            llm_client=self._llm_client,
            concurrency=max(1, debate_concurrency),
        )
        self.escalation_override = escalation_override
        self.max_debate_cost_usd = max_debate_cost_usd
        self.on_escalation = on_escalation
        self.on_verdict = on_verdict
        self.logger = logger or logging.getLogger(__name__)
        self._stats = JuryStats()

    async def classify(self, text: str) -> Verdict:
        start = time.perf_counter()
        primary = await self.classifier.classify(text)
        self._stats.total += 1

        should_escalate = self._should_escalate(primary) and bool(self.personas)

        if not should_escalate:
            self._stats.fast_path += 1
            return Verdict(
                label=primary.label,
                confidence=primary.confidence,
                reasoning="Classified by primary classifier with sufficient confidence.",
                was_escalated=False,
                primary_result=primary,
                debate_transcript=None,
                judge_strategy="primary_classifier",
                total_duration_ms=int((time.perf_counter() - start) * 1000),
                total_cost_usd=primary.cost_usd or 0.0,
            )

        self._stats.escalated += 1
        if self.on_escalation:
            self.on_escalation(text, primary)

        transcript = await self.debate_engine.debate(
            text=text,
            primary_result=primary,
            labels=self.classifier.labels,
            max_cost_usd=self.max_debate_cost_usd,
        )

        if self.max_debate_cost_usd is not None and transcript.total_cost_usd is not None:
            if transcript.total_cost_usd > self.max_debate_cost_usd:
                return Verdict(
                    label=primary.label,
                    confidence=primary.confidence,
                    reasoning=(
                        "Debate exceeded max_debate_cost_usd. "
                        "Returning primary classifier result."
                    ),
                    was_escalated=True,
                    primary_result=primary,
                    debate_transcript=transcript,
                    judge_strategy="cost_guard_primary_fallback",
                    total_duration_ms=int((time.perf_counter() - start) * 1000),
                    total_cost_usd=transcript.total_cost_usd,
                )

        verdict = await self.judge.judge(transcript, self.classifier.labels)
        verdict.was_escalated = True
        verdict.primary_result = primary
        verdict.debate_transcript = transcript
        verdict.total_duration_ms = int((time.perf_counter() - start) * 1000)

        if verdict.total_cost_usd is None:
            verdict.total_cost_usd = transcript.total_cost_usd

        if self.on_verdict:
            self.on_verdict(verdict)

        return verdict

    async def classify_batch(self, texts: list[str], concurrency: int = 10) -> list[Verdict]:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _classify(text: str) -> Verdict:
            async with sem:
                return await self.classify(text)

        return list(await asyncio.gather(*[_classify(text) for text in texts]))

    def _should_escalate(self, result: ClassificationResult) -> bool:
        if self.escalation_override is not None:
            return bool(self.escalation_override(result))
        return result.confidence < self.threshold

    @property
    def stats(self) -> JuryStats:
        return self._stats
