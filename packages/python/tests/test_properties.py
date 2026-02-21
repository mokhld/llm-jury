from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings, strategies as st

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import PersonaResponse


@given(
    confidence=st.floats(min_value=0.0, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
def test_escalation_boundary_property(confidence: float, threshold: float) -> None:
    """For any confidence and threshold: confidence >= threshold means no escalation."""
    clf = FunctionClassifier(fn=lambda t: ("safe", confidence), labels=["safe", "unsafe"])
    jury = Jury(classifier=clf, personas=[], confidence_threshold=threshold)
    result = ClassificationResult(label="safe", confidence=confidence)

    if confidence >= threshold:
        assert not jury._should_escalate(result)
    else:
        assert jury._should_escalate(result)


@given(st.lists(st.sampled_from(["safe", "unsafe"]), min_size=1, max_size=10))
def test_majority_vote_confidence_is_fraction(votes: list[str]) -> None:
    """Majority vote confidence should always be k/n for integer k, n."""
    responses = [
        PersonaResponse(
            persona_name=f"P{i}", label=v, confidence=0.8, reasoning="r", key_factors=[]
        )
        for i, v in enumerate(votes)
    ]
    transcript = DebateTranscript(
        input_text="test",
        primary_result=ClassificationResult("unknown", 0.5),
        rounds=[responses],
        duration_ms=0,
        total_tokens=0,
        total_cost_usd=0.0,
    )
    judge = MajorityVoteJudge()
    verdict = asyncio.run(judge.judge(transcript, ["safe", "unsafe"]))

    n = len(votes)
    assert verdict.confidence == pytest.approx(round(verdict.confidence * n) / n, abs=1e-9)


@given(st.lists(st.sampled_from(["safe", "unsafe"]), min_size=1, max_size=10))
@settings(max_examples=50)
def test_stats_invariant(classifications: list[str]) -> None:
    """fast_path + escalated should always equal total."""
    idx = [0]
    confs = [0.9 if c == "safe" else 0.3 for c in classifications]

    def classify_fn(text: str) -> tuple[str, float]:
        i = idx[0]
        idx[0] = i + 1
        return (classifications[i], confs[i])

    clf = FunctionClassifier(fn=classify_fn, labels=["safe", "unsafe"])
    jury = Jury(classifier=clf, personas=[], confidence_threshold=0.7)
    asyncio.run(jury.classify_batch([f"text-{i}" for i in range(len(classifications))]))

    assert jury.stats.fast_path + jury.stats.escalated == jury.stats.total
