from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import typer
from click.core import ParameterSource

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.calibration.optimizer import ThresholdCalibrator
from llm_jury.classifiers.base import Classifier
from llm_jury.classifiers.function_adapter import FunctionClassifier
from llm_jury.debate.engine import DebateConfig, DebateMode
from llm_jury.evaluation.evaluator import JuryEvaluator, TooManyEscalationsError
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.judges.weighted_vote import WeightedVoteJudge
from llm_jury.jury.core import Jury
from llm_jury.llm.client import LLMClient
from llm_jury.personas.base import Persona
from llm_jury.personas.registry import PersonaRegistry
from llm_jury.utils import json_serializable

app = typer.Typer(
    name="llm-jury",
    help="Confidence-driven escalation middleware for classifier edge cases.",
)


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
            fh.write(
                json.dumps(row, default=json_serializable, ensure_ascii=True) + "\n"
            )


def _parse_labels(raw: str | None, fallback: list[str] | None = None) -> list[str]:
    parsed = _explicit_labels(raw)
    if parsed:
        return parsed
    return fallback or ["safe", "unsafe"]


def _explicit_labels(raw: str | None) -> list[str] | None:
    """Labels passed with --labels, or None when the flag is absent or empty."""
    if not raw:
        return None
    parsed = [label.strip() for label in raw.split(",") if label.strip()]
    return parsed or None


# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------


def _check_unit_interval(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise typer.BadParameter(f"must be a number between 0 and 1, got {value}.")
    return value


def _check_positive_int(value: int) -> int:
    if value < 1:
        raise typer.BadParameter(f"must be an integer >= 1, got {value}.")
    return value


def _check_non_negative(value: float | None) -> float | None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise typer.BadParameter(f"must be a finite number >= 0, got {value}.")
    return value


def _check_non_negative_int(value: int | None) -> int | None:
    if value is not None and value < 0:
        raise typer.BadParameter(f"must be an integer >= 0, got {value}.")
    return value


def _parse_thresholds(raw: str | None, band_upper: float) -> list[float] | None:
    """Thresholds passed with --thresholds, or None when the flag is absent."""
    if raw is None:
        return None
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise typer.BadParameter(
                f"must be numbers between 0 and 1, got {text!r}.",
                param_hint="'--thresholds'",
            )
        if value > band_upper:
            raise typer.BadParameter(
                f"threshold {value} is above --band-upper {band_upper}; items at or "
                "above --band-upper are not debated, so their jury outcome is not "
                "measured.",
                param_hint="'--thresholds'",
            )
        values.append(value)
    if not values:
        raise typer.BadParameter(
            "needs at least one threshold.", param_hint="'--thresholds'"
        )
    return values


def _check_debate_mode(value: str) -> str:
    normalized = value.strip().lower()
    valid = [mode.value for mode in DebateMode]
    if normalized not in valid:
        raise typer.BadParameter(
            f"unsupported debate mode {value!r}. Use one of: {', '.join(valid)}.",
            param_hint="'--debate-mode'",
        )
    return normalized


def _format_row_numbers(row_numbers: list[int], limit: int = 5) -> str:
    shown = ", ".join(str(n) for n in row_numbers[:limit])
    if len(row_numbers) > limit:
        shown += f" (and {len(row_numbers) - limit} more)"
    return shown


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


def _select_judge(
    name: str, model: str | None = None, llm_client: LLMClient | None = None
):
    key = name.strip().lower()
    if key == "llm":
        from llm_jury.judges.llm_judge import LLMJudge

        return LLMJudge(model=model or DEFAULT_MODEL, llm_client=llm_client)
    if key == "majority":
        return MajorityVoteJudge()
    if key == "weighted":
        return WeightedVoteJudge()
    if key == "bayesian":
        from llm_jury.judges.bayesian import BayesianJudge

        return BayesianJudge()
    raise typer.BadParameter(f"Unsupported judge strategy: {name}")


def _prediction_confidence(value: object) -> float | None:
    """A stored prediction confidence as a float in [0, 1], or None if invalid."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _function_predictions(rows: list[dict]) -> dict[str, tuple[str, float]]:
    """Map each row's text to its stored ``(predicted_label, predicted_confidence)``.

    The ``function`` classifier replays predictions stored in the input file. It
    never reads the ground-truth ``label`` field, so calibration cannot score the
    ground truth against itself. Rows are numbered from 1 in error messages.
    """
    predictions: dict[str, tuple[str, float]] = {}
    invalid_rows: list[int] = []
    conflicting_rows: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        text = row.get("text")
        label = row.get("predicted_label")
        confidence = _prediction_confidence(row.get("predicted_confidence"))
        if (
            text is None
            or label is None
            or not str(label).strip()
            or confidence is None
        ):
            invalid_rows.append(row_number)
            continue
        key = str(text)
        prediction = (str(label), confidence)
        existing = predictions.setdefault(key, prediction)
        if existing != prediction:
            conflicting_rows.append(row_number)

    if invalid_rows:
        raise typer.BadParameter(
            "The 'function' classifier replays predictions stored in the input, so "
            "every row needs 'text', 'predicted_label' and 'predicted_confidence' "
            "(a number between 0 and 1). The ground-truth 'label' field is never "
            "used as a prediction. Missing or invalid in "
            f"{len(invalid_rows)} row(s): {_format_row_numbers(invalid_rows)}.",
            param_hint="'--classifier function'",
        )
    if conflicting_rows:
        raise typer.BadParameter(
            "The 'function' classifier looks up predictions by text, but "
            f"{len(conflicting_rows)} row(s) repeat an earlier text with a different "
            f"prediction: {_format_row_numbers(conflicting_rows)}.",
            param_hint="'--classifier function'",
        )
    return predictions


def _build_classifier(
    classifier_spec: str,
    labels: list[str],
    rows: list[dict],
    *,
    explicit_labels: list[str] | None = None,
    llm_client: LLMClient | None = None,
):
    """Build the primary classifier for a spec.

    ``labels`` is the label set for the run. ``explicit_labels`` is what the user
    passed with --labels (None when absent); the ``huggingface:`` spec uses it and
    otherwise takes label names from the model's scores. ``llm_client`` is used
    by the ``llm:`` spec (None means the default LiteLLM client).
    """
    spec = classifier_spec.strip()

    if spec == "function":
        predictions = _function_predictions(rows)
        classifier = FunctionClassifier(
            fn=lambda text, _pm=predictions: _pm[text],
            labels=labels,
        )
        return classifier, True

    if spec.startswith("llm:"):
        from llm_jury.classifiers.llm_classifier import LLMClassifier

        model_name = spec.split(":", 1)[1].strip()
        if not model_name:
            raise typer.BadParameter("classifier spec 'llm:' requires a model name")
        return (
            LLMClassifier(model=model_name, labels=labels, llm_client=llm_client),
            False,
        )

    if spec.startswith("huggingface:"):
        from llm_jury.classifiers.huggingface_adapter import HuggingFaceClassifier

        model_name = spec.split(":", 1)[1].strip()
        if not model_name:
            raise typer.BadParameter(
                "classifier spec 'huggingface:' requires a model name"
            )
        return (
            HuggingFaceClassifier(model_name=model_name, labels=explicit_labels),
            False,
        )

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
        mode=DebateMode(_check_debate_mode(debate_mode)),
        max_rounds=max_rounds,
        include_primary_result=not hide_primary_result,
        include_confidence=not hide_confidence,
    )


def _context_llm_client(ctx: typer.Context) -> LLMClient | None:
    """LLM client passed to :func:`main`, or None for the default LiteLLM client."""
    obj = ctx.obj
    return obj.get("llm_client") if isinstance(obj, dict) else None


def _build_jury(
    clf: Classifier,
    *,
    threshold: float,
    personas: str,
    persona_model: str | None,
    judge: str,
    judge_model: str | None,
    debate_mode: str,
    max_rounds: int,
    hide_primary_result: bool,
    hide_confidence: bool,
    max_debate_cost: float | None,
    debate_concurrency: int,
    llm_client: LLMClient | None,
) -> Jury:
    return Jury(
        classifier=clf,
        personas=_select_personas(personas, persona_model),
        confidence_threshold=threshold,
        judge=_select_judge(judge, judge_model, llm_client),
        debate_config=_build_debate_config(
            debate_mode, max_rounds, hide_primary_result, hide_confidence
        ),
        max_debate_cost_usd=max_debate_cost,
        debate_concurrency=debate_concurrency,
        llm_client=llm_client,
    )


def _load_labelled_rows(input: Path) -> tuple[list[dict], list[str], list[str]]:
    """Rows, texts and ground-truth labels of a calibration or evaluation file."""
    rows = _load_jsonl(input)
    if not rows:
        raise typer.BadParameter("Input JSONL is empty.")

    missing_labels = sum(1 for row in rows if "label" not in row)
    if missing_labels:
        raise typer.BadParameter(
            "Input requires a ground-truth 'label' field on every row. "
            f"Missing labels in {missing_labels} row(s)."
        )

    texts = [str(row.get("text", f"row-{idx}")) for idx, row in enumerate(rows)]
    labels_true = [str(row["label"]) for row in rows]
    return rows, texts, labels_true


# Options that only matter when the jury runs.
_JURY_OPTIONS = (
    ("personas", "--personas"),
    ("judge", "--judge"),
    ("judge_model", "--judge-model"),
    ("persona_model", "--persona-model"),
    ("debate_mode", "--debate-mode"),
    ("max_rounds", "--max-rounds"),
    ("max_debate_cost", "--max-debate-cost"),
    ("debate_concurrency", "--debate-concurrency"),
    ("hide_primary_result", "--hide-primary-result"),
    ("hide_confidence", "--hide-confidence"),
)


def _given_jury_options(ctx: typer.Context) -> list[str]:
    """Jury options the user set on the command line."""
    return [
        flag
        for name, flag in _JURY_OPTIONS
        if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def classify(
    ctx: typer.Context,
    input: Path = typer.Option(..., help="Input JSONL file"),
    output: Path = typer.Option(..., help="Output JSONL file"),
    classifier: str = typer.Option(
        "function", help="Classifier spec: function, llm:<model>, huggingface:<model>"
    ),
    personas: str = typer.Option("content_moderation", help="Persona set name"),
    labels: str | None = typer.Option(None, help="Comma-separated label list"),
    judge: str = typer.Option(
        "llm", help="Judge strategy: llm, majority, weighted, bayesian"
    ),
    judge_model: str = typer.Option(DEFAULT_MODEL, help="Model for LLM judge"),
    persona_model: str | None = typer.Option(
        None, help="Override model for all personas"
    ),
    threshold: float = typer.Option(
        0.7,
        help="Confidence threshold for escalation (0 to 1)",
        callback=_check_unit_interval,
    ),
    concurrency: int = typer.Option(
        10, help="Batch concurrency", callback=_check_positive_int
    ),
    debate_mode: str = typer.Option(
        "independent",
        help="Debate mode: independent, sequential, deliberation, adversarial",
        callback=_check_debate_mode,
    ),
    max_rounds: int = typer.Option(
        1, help="Max deliberation rounds", callback=_check_positive_int
    ),
    max_debate_cost: float | None = typer.Option(
        None, help="Max debate cost in USD", callback=_check_non_negative
    ),
    debate_concurrency: int = typer.Option(
        5, help="Debate persona concurrency", callback=_check_positive_int
    ),
    hide_primary_result: bool = typer.Option(
        False, help="Hide primary result from personas"
    ),
    hide_confidence: bool = typer.Option(False, help="Hide confidence from personas"),
) -> None:
    """Classify JSONL texts with confidence-based jury escalation."""
    llm_client = _context_llm_client(ctx)
    input_rows = _load_jsonl(input)
    texts = [str(row.get("text", "")) for row in input_rows]
    parsed_labels = _parse_labels(labels, fallback=["safe", "unsafe"])
    clf, is_mock = _build_classifier(
        classifier,
        parsed_labels,
        input_rows,
        explicit_labels=_explicit_labels(labels),
        llm_client=llm_client,
    )

    jury_instance = _build_jury(
        clf,
        threshold=threshold,
        personas=personas,
        persona_model=persona_model,
        judge=judge,
        judge_model=judge_model,
        debate_mode=debate_mode,
        max_rounds=max_rounds,
        hide_primary_result=hide_primary_result,
        hide_confidence=hide_confidence,
        max_debate_cost=max_debate_cost,
        debate_concurrency=debate_concurrency,
        llm_client=llm_client,
    )

    effective_concurrency = 1 if is_mock else concurrency
    results = asyncio.run(
        jury_instance.classify_batch(
            texts, concurrency=effective_concurrency, return_exceptions=True
        )
    )

    rows: list[dict] = []
    failures = 0
    for text, result in zip(texts, results, strict=True):
        if isinstance(result, BaseException):
            failures += 1
            rows.append({"text": text, "error": f"{type(result).__name__}: {result}"})
        else:
            rows.append(result.to_dict())

    _write_jsonl(output, rows)
    typer.echo(f"Wrote {len(rows)} verdict(s) to {output}")
    if failures:
        typer.echo(
            f"Warning: {failures} of {len(rows)} row(s) failed; "
            "failed rows contain an 'error' field instead of a verdict.",
            err=True,
        )
        if failures == len(rows):
            raise typer.Exit(code=1)


@app.command()
def calibrate(
    ctx: typer.Context,
    input: Path = typer.Option(
        ..., help="Input JSONL file with ground-truth 'label' field"
    ),
    classifier: str = typer.Option("function", help="Classifier spec"),
    personas: str = typer.Option("content_moderation", help="Persona set name"),
    labels: str | None = typer.Option(None, help="Comma-separated label list"),
    judge: str = typer.Option("llm", help="Judge strategy"),
    judge_model: str = typer.Option(DEFAULT_MODEL, help="Model for LLM judge"),
    persona_model: str | None = typer.Option(
        None, help="Override model for all personas"
    ),
    initial_threshold: float = typer.Option(
        0.7, help="Starting threshold (0 to 1)", callback=_check_unit_interval
    ),
    error_cost: float = typer.Option(
        10.0, help="Cost per classification error", callback=_check_non_negative
    ),
    escalation_cost: float | None = typer.Option(
        None,
        help=(
            "Cost per escalation (default 0.05; with --use-jury, the measured mean "
            "debate cost)"
        ),
        callback=_check_non_negative,
    ),
    use_jury: bool = typer.Option(
        False,
        help=(
            "Run the jury on every item below the highest threshold and pick the "
            "threshold from its measured outcomes (makes LLM calls)"
        ),
    ),
    debate_mode: str = typer.Option(
        "independent",
        help="Debate mode: independent, sequential, deliberation, adversarial",
        callback=_check_debate_mode,
    ),
    max_rounds: int = typer.Option(
        1, help="Max deliberation rounds", callback=_check_positive_int
    ),
    max_debate_cost: float | None = typer.Option(
        None, help="Max debate cost in USD", callback=_check_non_negative
    ),
    debate_concurrency: int = typer.Option(
        5, help="Debate persona concurrency", callback=_check_positive_int
    ),
    hide_primary_result: bool = typer.Option(
        False, help="Hide primary result from personas"
    ),
    hide_confidence: bool = typer.Option(False, help="Hide confidence from personas"),
) -> None:
    """Calibrate the optimal confidence threshold from labelled data."""
    llm_client = _context_llm_client(ctx)
    rows, texts, labels_true = _load_labelled_rows(input)
    parsed_labels = _parse_labels(labels, fallback=sorted(set(labels_true)))
    clf, _ = _build_classifier(
        classifier,
        parsed_labels,
        rows,
        explicit_labels=_explicit_labels(labels),
        llm_client=llm_client,
    )

    if not use_jury:
        ignored = _given_jury_options(ctx)
        if ignored:
            typer.echo(
                "Note: calibrate without --use-jury never runs the jury, so "
                f"{', '.join(ignored)} had no effect. Pass --use-jury to measure "
                "the jury's outcomes.",
                err=True,
            )

    jury_instance = _build_jury(
        clf,
        threshold=initial_threshold,
        personas=personas,
        persona_model=persona_model,
        judge=judge,
        judge_model=judge_model,
        debate_mode=debate_mode,
        max_rounds=max_rounds,
        hide_primary_result=hide_primary_result,
        hide_confidence=hide_confidence,
        max_debate_cost=max_debate_cost,
        debate_concurrency=debate_concurrency,
        llm_client=llm_client,
    )

    calibrator = ThresholdCalibrator(jury_instance)
    asyncio.run(
        calibrator.calibrate(
            texts=texts,
            labels=labels_true,
            error_cost=error_cost,
            escalation_cost=escalation_cost,
            use_jury=use_jury,
        )
    )
    report = calibrator.calibration_report()
    typer.echo(json.dumps(report, ensure_ascii=True))


@app.command("eval")
def eval_command(
    ctx: typer.Context,
    input: Path = typer.Option(
        ..., help="Input JSONL file with ground-truth 'label' field"
    ),
    output: Path | None = typer.Option(
        None, help="Also write the full report, with per-item results, to this file"
    ),
    classifier: str = typer.Option(
        "function", help="Classifier spec: function, llm:<model>, huggingface:<model>"
    ),
    personas: str = typer.Option("content_moderation", help="Persona set name"),
    labels: str | None = typer.Option(None, help="Comma-separated label list"),
    judge: str = typer.Option(
        "llm", help="Judge strategy: llm, majority, weighted, bayesian"
    ),
    judge_model: str = typer.Option(DEFAULT_MODEL, help="Model for LLM judge"),
    persona_model: str | None = typer.Option(
        None, help="Override model for all personas"
    ),
    band_upper: float = typer.Option(
        0.95,
        help="Debate every item whose primary confidence is below this (0 to 1)",
        callback=_check_unit_interval,
    ),
    max_escalations: int | None = typer.Option(
        None,
        help="Stop before any debate if more items than this would be debated",
        callback=_check_non_negative_int,
    ),
    thresholds: str | None = typer.Option(
        None,
        help=(
            "Comma-separated thresholds to sweep, each <= --band-upper "
            "(default 0.5,0.55,...,0.95 up to --band-upper)"
        ),
    ),
    error_cost: float = typer.Option(
        10.0, help="Cost per wrong final label", callback=_check_non_negative
    ),
    escalation_cost: float | None = typer.Option(
        None,
        help="Cost per escalation (default: the measured mean debate cost)",
        callback=_check_non_negative,
    ),
    concurrency: int = typer.Option(
        5, help="Items classified or debated at once", callback=_check_positive_int
    ),
    debate_mode: str = typer.Option(
        "independent",
        help="Debate mode: independent, sequential, deliberation, adversarial",
        callback=_check_debate_mode,
    ),
    max_rounds: int = typer.Option(
        1, help="Max deliberation rounds", callback=_check_positive_int
    ),
    max_debate_cost: float | None = typer.Option(
        None, help="Max debate cost in USD", callback=_check_non_negative
    ),
    debate_concurrency: int = typer.Option(
        5, help="Debate persona concurrency", callback=_check_positive_int
    ),
    hide_primary_result: bool = typer.Option(
        False, help="Hide primary result from personas"
    ),
    hide_confidence: bool = typer.Option(False, help="Hide confidence from personas"),
) -> None:
    """Measure the jury against the primary classifier on labelled data."""
    llm_client = _context_llm_client(ctx)
    sweep_thresholds = _parse_thresholds(thresholds, band_upper)
    rows, texts, labels_true = _load_labelled_rows(input)
    parsed_labels = _parse_labels(labels, fallback=sorted(set(labels_true)))
    clf, _ = _build_classifier(
        classifier,
        parsed_labels,
        rows,
        explicit_labels=_explicit_labels(labels),
        llm_client=llm_client,
    )
    jury_instance = _build_jury(
        clf,
        threshold=0.7,
        personas=personas,
        persona_model=persona_model,
        judge=judge,
        judge_model=judge_model,
        debate_mode=debate_mode,
        max_rounds=max_rounds,
        hide_primary_result=hide_primary_result,
        hide_confidence=hide_confidence,
        max_debate_cost=max_debate_cost,
        debate_concurrency=debate_concurrency,
        llm_client=llm_client,
    )

    try:
        report = asyncio.run(
            JuryEvaluator(jury_instance).evaluate(
                texts,
                labels_true,
                band_upper=band_upper,
                max_escalations=max_escalations,
                concurrency=concurrency,
            )
        )
    except TooManyEscalationsError as exc:
        raise typer.BadParameter(str(exc), param_hint="'--max-escalations'") from exc

    sweep = report.threshold_sweep(
        sweep_thresholds, error_cost=error_cost, escalation_cost=escalation_cost
    )
    best = report.best_threshold(
        error_cost=error_cost,
        escalation_cost=escalation_cost,
        thresholds=sweep_thresholds,
    )
    summary = report.summary()
    typer.echo(
        json.dumps(
            {"best_threshold": best, "summary": summary, "sweep": sweep},
            ensure_ascii=True,
        )
    )
    if output is not None:
        full = {
            "best_threshold": best,
            "band_upper": report.band_upper,
            "summary": summary,
            "sweep": sweep,
            "items": [item.to_dict() for item in report.items],
        }
        output.write_text(
            json.dumps(full, indent=2, default=json_serializable, ensure_ascii=True)
            + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None, *, llm_client: LLMClient | None = None) -> None:
    """Run the CLI. ``llm_client`` replaces the default LiteLLM client for every
    LLM call the command makes (personas, LLM judge, ``llm:`` classifier), which
    lets tests and embedding code run the commands without network access."""
    obj = {"llm_client": llm_client}
    if argv is not None:
        app(standalone_mode=False, args=argv, obj=obj)
    else:
        app(obj=obj)


if __name__ == "__main__":
    main()
