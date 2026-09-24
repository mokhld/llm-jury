from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

from llm_jury.classifiers.base import ClassificationResult, Classifier
from llm_jury.debate.engine import (
    DebateConfig,
    DebateEngine,
    DebateMode,
    exceeds_cost_cap,
    guard_spend,
)
from llm_jury.judges.base import JudgeStrategy, Verdict
from llm_jury.judges.llm_judge import LLMJudge
from llm_jury.llm.client import LiteLLMClient, LLMClient
from llm_jury.personas.base import Persona
from llm_jury.utils import add_costs, is_finite_number


def _escalated_total_cost(
    primary_cost_usd: float | None, debate_cost_usd: float | None
) -> float | None:
    """Total cost of an escalated verdict: primary call plus debate (and judge).

    Either part being unknown makes the total unknown. Summing only the known
    part would report a debate whose calls were all unpriced as free.
    """
    if primary_cost_usd is None or debate_cost_usd is None:
        return None
    return primary_cost_usd + debate_cost_usd


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
        estimated_cost_per_persona_usd: float = 0.01,
        on_escalation: Callable[[str, ClassificationResult], None] | None = None,
        on_cost_estimate: Callable[[float, str], bool | None] | None = None,
        on_verdict: Callable[[Verdict], None] | None = None,
        logger: logging.Logger | None = None,
        llm_client: LLMClient | None = None,
        debate_concurrency: int = 5,
    ) -> None:
        if not is_finite_number(confidence_threshold) or not (
            0.0 <= confidence_threshold <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must be a finite number in [0, 1], "
                f"got {confidence_threshold!r}"
            )
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
        # Estimated cost of one LLM call (persona, summariser or judge). Used
        # for the pre-flight estimate and, by the cost guard, for every call
        # whose client reported no cost.
        self.estimated_cost_per_persona_usd = max(0.0, estimated_cost_per_persona_usd)
        self.on_escalation = on_escalation
        self.on_cost_estimate = on_cost_estimate
        self.on_verdict = on_verdict
        self.logger = logger or logging.getLogger(__name__)
        self._stats = JuryStats()

    @property
    def estimated_max_debate_cost_usd(self) -> float:
        """Upper-bound estimate of one escalation's LLM spend.

        Counts every call the debate and judge can make: each persona in
        each round, the summariser (deliberation mode) and the judge (when it
        is an ``LLMJudge``), priced at ``estimated_cost_per_persona_usd``.
        """
        deliberation = self.debate_config.mode == DebateMode.DELIBERATION
        rounds = max(1, self.debate_config.max_rounds) if deliberation else 1
        persona_calls = len(self.personas) * rounds
        summariser_calls = 1 if deliberation else 0
        judge_calls = 1 if isinstance(self.judge, LLMJudge) else 0
        return self.estimated_cost_per_persona_usd * (
            persona_calls + summariser_calls + judge_calls
        )

    def _finish(self, verdict: Verdict) -> Verdict:
        """Single exit for every verdict ``classify`` returns."""
        if self.on_verdict is not None:
            self.on_verdict(verdict)
        return verdict

    async def classify(self, text: str) -> Verdict:
        start = time.perf_counter()
        primary = await self.classifier.classify(text)
        self._stats.total += 1

        should_escalate = self._should_escalate(primary) and bool(self.personas)

        if not should_escalate:
            self._stats.fast_path += 1
            return self._finish(
                Verdict(
                    label=primary.label,
                    confidence=primary.confidence,
                    reasoning="Classified by primary classifier with sufficient confidence.",
                    was_escalated=False,
                    primary_result=primary,
                    debate_transcript=None,
                    judge_strategy="primary_classifier",
                    total_duration_ms=int((time.perf_counter() - start) * 1000),
                    total_cost_usd=primary.cost_usd,
                )
            )

        self._stats.escalated += 1
        return await self._escalate(text, primary, start)

    async def escalate(self, text: str, primary: ClassificationResult) -> Verdict:
        """Run the escalation branch for a primary result you already have.

        Skips the primary classifier and the threshold check, then does what
        ``classify`` does for an escalated item: fires ``on_escalation``,
        applies the ``on_cost_estimate`` gate and the cost guards, runs the
        debate and the judge, and fires ``on_verdict``. It does not update
        ``stats``. ``total_duration_ms`` counts from this call, so it leaves
        out the primary classifier's time. ``classify`` runs the same code for
        every item it escalates.

        Raises ``ValueError`` when the jury has no personas, because there is
        no one to debate.
        """
        if not self.personas:
            raise ValueError(
                "Jury.escalate needs at least one persona; this jury has none."
            )
        return await self._escalate(text, primary, time.perf_counter())

    async def _escalate(
        self, text: str, primary: ClassificationResult, start: float
    ) -> Verdict:
        if self.on_escalation:
            self.on_escalation(text, primary)

        # F4: optional user-supplied pre-debate cost gate. Fires before
        # the hardcoded `max_debate_cost_usd` guard so user logic can
        # short-circuit on policy beyond a fixed cap (per-tenant
        # budgets, time-of-day, etc.). Returning False (or any falsy
        # non-None value) skips the debate. Returning True or None
        # proceeds.
        if self.on_cost_estimate is not None:
            decision = self.on_cost_estimate(self.estimated_max_debate_cost_usd, text)
            if decision is False:
                self.logger.info(
                    "[llm-jury] skipping debate: on_cost_estimate returned False "
                    "for estimate %.4f USD",
                    self.estimated_max_debate_cost_usd,
                )
                return self._finish(
                    Verdict(
                        label=primary.label,
                        confidence=primary.confidence,
                        reasoning=(
                            "Debate skipped: on_cost_estimate callback returned "
                            "False. Returning primary classifier result."
                        ),
                        was_escalated=True,
                        primary_result=primary,
                        debate_transcript=None,
                        judge_strategy="cost_guard_user_override",
                        total_duration_ms=int((time.perf_counter() - start) * 1000),
                        total_cost_usd=add_costs(primary.cost_usd),
                    )
                )

        if exceeds_cost_cap(
            self.estimated_max_debate_cost_usd, self.max_debate_cost_usd
        ):
            self.logger.warning(
                "[llm-jury] skipping debate: estimated cost %.4f USD exceeds "
                "max_debate_cost_usd %.4f USD",
                self.estimated_max_debate_cost_usd,
                self.max_debate_cost_usd,
            )
            return self._finish(
                Verdict(
                    label=primary.label,
                    confidence=primary.confidence,
                    reasoning=(
                        "Debate skipped: estimated cost "
                        f"({self.estimated_max_debate_cost_usd:.4f} USD) exceeds "
                        f"max_debate_cost_usd ({self.max_debate_cost_usd:.4f} USD). "
                        "Returning primary classifier result."
                    ),
                    was_escalated=True,
                    primary_result=primary,
                    debate_transcript=None,
                    judge_strategy="cost_guard_pre_flight",
                    total_duration_ms=int((time.perf_counter() - start) * 1000),
                    total_cost_usd=add_costs(primary.cost_usd),
                )
            )

        transcript = await self.debate_engine.debate(
            text=text,
            primary_result=primary,
            labels=self.classifier.labels,
            max_cost_usd=self.max_debate_cost_usd,
            estimated_cost_per_call_usd=self.estimated_cost_per_persona_usd,
        )

        debate_spend = guard_spend(
            transcript.total_cost_usd,
            transcript.unpriced_calls,
            self.estimated_cost_per_persona_usd,
        )
        if exceeds_cost_cap(debate_spend, self.max_debate_cost_usd):
            return self._finish(
                Verdict(
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
                    total_cost_usd=_escalated_total_cost(
                        primary.cost_usd, transcript.total_cost_usd
                    ),
                    persona_failures=transcript.persona_failures,
                )
            )

        verdict = await self.judge.judge(transcript, self.classifier.labels)

        # Jury is authoritative for `was_escalated` and `persona_failures`:
        # it KNOWS this code path is the escalation branch and it holds the
        # transcript, so judges can't override either.
        verdict.was_escalated = True
        verdict.persona_failures = transcript.persona_failures
        if verdict.persona_failures:
            self.logger.warning(
                "[llm-jury] verdict is degraded: %d persona call(s) failed "
                "during the debate",
                verdict.persona_failures,
            )

        # Backfill fields the judge may have left at their default/unset value.
        # Custom judges that populated these intentionally are respected.
        if verdict.primary_result is None:  # type: ignore[unreachable]
            verdict.primary_result = primary
        if verdict.debate_transcript is None:
            verdict.debate_transcript = transcript
        if not verdict.total_duration_ms:
            verdict.total_duration_ms = int((time.perf_counter() - start) * 1000)
        # The judge reports debate + judge cost; the verdict total also
        # includes the primary classifier call.
        debate_and_judge_cost = (
            verdict.total_cost_usd
            if verdict.total_cost_usd is not None
            else transcript.total_cost_usd
        )
        verdict.total_cost_usd = _escalated_total_cost(
            primary.cost_usd, debate_and_judge_cost
        )

        return self._finish(verdict)

    @overload
    async def classify_batch(
        self,
        texts: list[str],
        concurrency: int = 10,
        return_exceptions: Literal[False] = False,
    ) -> list[Verdict]: ...

    @overload
    async def classify_batch(
        self,
        texts: list[str],
        concurrency: int = 10,
        *,
        return_exceptions: Literal[True],
    ) -> list[Verdict | BaseException]: ...

    async def classify_batch(
        self,
        texts: list[str],
        concurrency: int = 10,
        return_exceptions: bool = False,
    ) -> list[Verdict] | list[Verdict | BaseException]:
        """Classify many texts concurrently.

        With ``return_exceptions=False`` (default) the first failing text
        raises and the whole batch is lost. No new ``classify`` call starts
        after that failure, and calls still queued or in flight are
        cancelled, so a failed batch stops spending. Pass
        ``return_exceptions=True`` to receive the exception object in that
        text's slot instead, so one bad row cannot discard the verdicts (and
        spend) of the rows that succeeded.
        """
        sem = asyncio.Semaphore(max(1, concurrency))
        aborted = False

        async def _classify(text: str) -> Verdict:
            nonlocal aborted
            async with sem:
                if aborted:
                    raise asyncio.CancelledError()
                try:
                    return await self.classify(text)
                except BaseException:
                    if not return_exceptions:
                        aborted = True
                    raise

        if return_exceptions:
            return list(
                await asyncio.gather(
                    *[_classify(text) for text in texts],
                    return_exceptions=True,
                )
            )

        tasks = [asyncio.ensure_future(_classify(text)) for text in texts]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            aborted = True
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for the cancelled tasks to unwind so none outlives the call.
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _should_escalate(self, result: ClassificationResult) -> bool:
        if self.escalation_override is not None:
            return bool(self.escalation_override(result))
        # A missing or non-finite confidence (None, NaN, inf) cannot vouch
        # for the primary label, so it always escalates.
        if not is_finite_number(result.confidence):
            return True
        return result.confidence < self.threshold

    @property
    def stats(self) -> JuryStats:
        return self._stats
