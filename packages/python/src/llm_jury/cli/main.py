from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.calibration.optimizer import ThresholdCalibrator
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.judges.weighted_vote import WeightedVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.personas.base import Persona
from llm_jury.personas.registry import PersonaRegistry

app = typer.Typer(name="llm-jury", help="Confidence-driven escalation middleware for classifier edge cases.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _parse_labels(raw: str | None, fallback: list[str] | None = None) -> list[str]:
    if raw:
        return [label.strip() for label in raw.split(",") if label.strip()]
    return fallback or ["safe", "unsafe"]


def _apply_persona_model(personas: list[Persona], model: str | None) -> list[Persona]:
    if not model:
        return personas
    return [
        Persona(
            name=p.name,
            role=p.role,
            system_prompt=p.system_prompt,
            model=model,
            temperature=p.temperature,
            known_bias=p.known_bias,
        )
        for p in personas
    ]


def _select_personas(name: str, model: str | None = None) -> list[Persona]:
    key = name.strip().lower()
    registry_map = {
        "content_moderation": PersonaRegistry.content_moderation,
        "legal_compliance": PersonaRegistry.legal_compliance,
        "medical_triage": PersonaRegistry.medical_triage,
        "financial_compliance": PersonaRegistry.financial_compliance,
    }
    factory = registry_map.get(key)
    if factory is None:
        raise typer.BadParameter(f"Unsupported personas set: {name}")
    return _apply_persona_model(factory(), model)


def _select_judge(name: str, model: str | None = None):
    key = name.strip().lower()
    if key == "llm":
        from llm_jury.judges.llm_judge import LLMJudge
        return LLMJudge(model=model or DEFAULT_MODEL)
    if key == "majority":
        return MajorityVoteJudge()
    if key == "weighted":
        return WeightedVoteJudge()
    if key == "bayesian":
        from llm_jury.judges.bayesian import BayesianJudge
        return BayesianJudge()
    raise typer.BadParameter(f"Unsupported judge strategy: {name}")


def _build_classifier(classifier_spec: str, labels: list[str], rows: list[dict]):
    spec = classifier_spec.strip()

    if spec == "function":
        prediction_map: dict[str, tuple[str, float]] = {}
        for idx, row in enumerate(rows):
            text = str(row.get("text", f"row-{idx}"))
            pred_label = str(row.get("predicted_label", row.get("label", labels[0])))
            pred_conf = float(row.get("predicted_confidence", 0.95))
            prediction_map[text] = (pred_label, pred_conf)

        default_result = (labels[0], 0.95)
        classifier = FunctionClassifier(
            fn=lambda text, _pm=prediction_map, _d=default_result: _pm.get(text, _d),
            labels=labels,
        )
        return classifier, True

    if spec.startswith("llm:"):
        from llm_jury.classifiers.llm_classifier import LLMClassifier
        model_name = spec.split(":", 1)[1].strip()
        if not model_name:
            raise typer.BadParameter("classifier spec 'llm:' requires a model name")
        return LLMClassifier(model=model_name, labels=labels), False

    if spec.startswith("huggingface:"):
        from llm_jury.classifiers.huggingface_adapter import HuggingFaceClassifier
        model_name = spec.split(":", 1)[1].strip()
        if not model_name:
            raise typer.BadParameter("classifier spec 'huggingface:' requires a model name")
        return HuggingFaceClassifier(model_name=model_name), False

    raise typer.BadParameter(
        "Unsupported classifier spec. Use one of: function, llm:<model>, huggingface:<model>"
    )


def _build_debate_config(
    debate_mode: str,
    max_rounds: int,
    hide_primary_result: bool,
    hide_confidence: bool,
) -> DebateConfig:
    return DebateConfig(
        mode=DebateMode(debate_mode),
        max_rounds=max_rounds,
        include_primary_result=not hide_primary_result,
        include_confidence=not hide_confidence,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def classify(
    input: Path = typer.Option(..., help="Input JSONL file"),
    output: Path = typer.Option(..., help="Output JSONL file"),
    classifier: str = typer.Option("function", help="Classifier spec: function, llm:<model>, huggingface:<model>"),
    personas: str = typer.Option("content_moderation", help="Persona set name"),
    labels: Optional[str] = typer.Option(None, help="Comma-separated label list"),
    judge: str = typer.Option("llm", help="Judge strategy: llm, majority, weighted, bayesian"),
    judge_model: str = typer.Option(DEFAULT_MODEL, help="Model for LLM judge"),
    persona_model: Optional[str] = typer.Option(None, help="Override model for all personas"),
    threshold: float = typer.Option(0.7, help="Confidence threshold for escalation"),
    concurrency: int = typer.Option(10, help="Batch concurrency"),
    debate_mode: str = typer.Option("independent", help="Debate mode"),
    max_rounds: int = typer.Option(1, help="Max deliberation rounds"),
    max_debate_cost: Optional[float] = typer.Option(None, help="Max debate cost in USD"),
    debate_concurrency: int = typer.Option(5, help="Debate persona concurrency"),
    hide_primary_result: bool = typer.Option(False, help="Hide primary result from personas"),
    hide_confidence: bool = typer.Option(False, help="Hide confidence from personas"),
) -> None:
    """Classify JSONL texts with confidence-based jury escalation."""
    input_rows = _load_jsonl(input)
    texts = [str(row.get("text", "")) for row in input_rows]
    parsed_labels = _parse_labels(labels, fallback=["safe", "unsafe"])
    clf, is_mock = _build_classifier(classifier, parsed_labels, input_rows)

    jury_instance = Jury(
        classifier=clf,
        personas=_select_personas(personas, persona_model),
        confidence_threshold=threshold,
        judge=_select_judge(judge, judge_model),
        debate_config=_build_debate_config(debate_mode, max_rounds, hide_primary_result, hide_confidence),
        max_debate_cost_usd=max_debate_cost,
        debate_concurrency=debate_concurrency,
    )

    effective_concurrency = 1 if is_mock else concurrency
    verdicts = asyncio.run(jury_instance.classify_batch(texts, concurrency=effective_concurrency))
    _write_jsonl(output, [v.to_dict() for v in verdicts])
    typer.echo(f"Wrote {len(verdicts)} verdict(s) to {output}")


@app.command()
def calibrate(
    input: Path = typer.Option(..., help="Input JSONL file with ground-truth 'label' field"),
    classifier: str = typer.Option("function", help="Classifier spec"),
    personas: str = typer.Option("content_moderation", help="Persona set name"),
    labels: Optional[str] = typer.Option(None, help="Comma-separated label list"),
    judge: str = typer.Option("llm", help="Judge strategy"),
    judge_model: str = typer.Option(DEFAULT_MODEL, help="Model for LLM judge"),
    persona_model: Optional[str] = typer.Option(None, help="Override model for all personas"),
    initial_threshold: float = typer.Option(0.7, help="Starting threshold"),
    error_cost: float = typer.Option(10.0, help="Cost per classification error"),
    escalation_cost: float = typer.Option(0.05, help="Cost per escalation"),
    debate_mode: str = typer.Option("independent", help="Debate mode"),
    max_rounds: int = typer.Option(1, help="Max deliberation rounds"),
    max_debate_cost: Optional[float] = typer.Option(None, help="Max debate cost in USD"),
    debate_concurrency: int = typer.Option(5, help="Debate persona concurrency"),
    hide_primary_result: bool = typer.Option(False, help="Hide primary result from personas"),
    hide_confidence: bool = typer.Option(False, help="Hide confidence from personas"),
) -> None:
    """Calibrate the optimal confidence threshold from labelled data."""
    rows = _load_jsonl(input)
    if not rows:
        raise typer.BadParameter("Input JSONL is empty.")

    missing_labels = sum(1 for row in rows if "label" not in row)
    if missing_labels:
        raise typer.BadParameter(
            f"Calibration input requires a ground-truth 'label' field on every row. "
            f"Missing labels in {missing_labels} row(s)."
        )

    texts = [str(row.get("text", f"row-{idx}")) for idx, row in enumerate(rows)]
    labels_true = [str(row["label"]) for row in rows]
    parsed_labels = _parse_labels(labels, fallback=sorted(set(labels_true)))
    clf, _ = _build_classifier(classifier, parsed_labels, rows)

    jury_instance = Jury(
        classifier=clf,
        personas=_select_personas(personas, persona_model),
        confidence_threshold=initial_threshold,
        judge=_select_judge(judge, judge_model),
        debate_config=_build_debate_config(debate_mode, max_rounds, hide_primary_result, hide_confidence),
        max_debate_cost_usd=max_debate_cost,
        debate_concurrency=debate_concurrency,
    )

    calibrator = ThresholdCalibrator(jury_instance)
    best = asyncio.run(
        calibrator.calibrate(
            texts=texts,
            labels=labels_true,
            error_cost=error_cost,
            escalation_cost=escalation_cost,
        )
    )
    report = calibrator.calibration_report()
    typer.echo(json.dumps(report, ensure_ascii=True))


def main(argv: list[str] | None = None) -> None:
    if argv is not None:
        app(standalone_mode=False, args=argv)
    else:
        app()


if __name__ == "__main__":
    main()
