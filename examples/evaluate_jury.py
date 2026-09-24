"""Measure whether the jury beats the primary classifier on labelled data.

Runs offline: the personas use a stub LLM client that answers from a lookup
table, so no API key is needed. In production, drop ``llm_client`` and the jury
uses its default LiteLLM client (and real money, so set ``max_escalations``).
"""

from __future__ import annotations

import asyncio
import json

from llm_jury import DebateConfig, DebateMode, Jury, JuryEvaluator, Persona
from llm_jury.classifiers import FunctionClassifier
from llm_jury.judges import MajorityVoteJudge

# text, true label, primary label, primary confidence, what the stub jury answers
DATA = [
    ("Thanks for the quick reply, see you Monday.", "safe", "safe", 0.98, "safe"),
    ("Buy cheap followers now, limited offer!!!", "unsafe", "unsafe", 0.97, "unsafe"),
    ("I will find out where you live.", "unsafe", "safe", 0.62, "unsafe"),
    ("This movie absolutely killed it.", "safe", "unsafe", 0.58, "safe"),
    ("Great, another Monday. Kill me now.", "safe", "unsafe", 0.66, "safe"),
    ("Send me your password to verify your account.", "unsafe", "unsafe", 0.81, "unsafe"),
    ("The recipe calls for a pinch of salt.", "safe", "safe", 0.93, "safe"),
    ("You people are the worst, get out.", "unsafe", "unsafe", 0.72, "unsafe"),
    ("Meet me behind the gym after school.", "safe", "unsafe", 0.55, "unsafe"),
    ("Nice shot, you really destroyed them.", "safe", "safe", 0.77, "unsafe"),
    ("Click here to claim your prize.", "unsafe", "safe", 0.90, "unsafe"),
    ("Happy birthday, have a great one!", "safe", "safe", 0.99, "safe"),
]


class StubLLMClient:
    """Answers every persona call from a lookup table, at a made-up cost."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> dict:
        label = next(ans for text, ans in self.answers.items() if text in prompt)
        content = {"label": label, "confidence": 0.8, "reasoning": "stub"}
        return {"content": json.dumps(content), "tokens": 50, "cost_usd": 0.0004}


async def main() -> None:
    texts = [row[0] for row in DATA]
    labels = [row[1] for row in DATA]
    predictions = {row[0]: (row[2], row[3]) for row in DATA}

    jury = Jury(
        classifier=FunctionClassifier(
            fn=lambda text: predictions[text], labels=["safe", "unsafe"]
        ),
        personas=[
            Persona(name=name, role=name, system_prompt=f"You are the {name}.")
            for name in ("policy analyst", "context reader", "harm assessor")
        ],
        judge=MajorityVoteJudge(),
        debate_config=DebateConfig(mode=DebateMode.INDEPENDENT),
        llm_client=StubLLMClient({row[0]: row[4] for row in DATA}),
    )

    report = await JuryEvaluator(jury).evaluate(
        texts, labels, band_upper=0.95, max_escalations=50
    )
    summary = report.summary()
    print(f"Items: {summary['n']}, debated: {summary['debated']}")
    print(f"Primary accuracy:             {summary['primary_accuracy']:.2f}")
    print(f"Primary accuracy on debated:  {summary['primary_accuracy_on_debated']:.2f}")
    print(f"Jury accuracy on debated:     {summary['jury_accuracy_on_debated']:.2f}")
    print(f"Flips helped / hurt:          {summary['flips_helped']} / {summary['flips_hurt']}")
    print(f"Mean debate cost:             ${summary['mean_debate_cost_usd']:.4f}")
    print()

    for row in report.threshold_sweep(error_cost=10.0):
        print(
            f"  threshold={row['threshold']:.2f}  "
            f"escalation_rate={row['escalation_rate']:.2f}  "
            f"system_accuracy={row['system_accuracy']:.2f}  "
            f"errors={row['errors']}  "
            f"cost={row['total_cost']:.2f}"
        )
    print()
    print(f"Best threshold: {report.best_threshold(error_cost=10.0)}")


if __name__ == "__main__":
    asyncio.run(main())
