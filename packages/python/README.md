# llm-jury

**When your classifier is uncertain, let a configurable jury of LLM personas debate and return an auditable verdict.**

[![PyPI](https://img.shields.io/pypi/v/llm-jury)](https://pypi.org/project/llm-jury/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

## Overview

`llm-jury` is an SDK, not a hosted API. Your app imports it directly:

```python
from llm_jury import Jury, PersonaRegistry
```

It wraps a classifier returning `(label, confidence)` and adds confidence-based escalation:

1. Run primary classifier (fast path)
2. Return directly when confidence is high
3. Escalate low-confidence cases to persona debate
4. Consolidate with a judge strategy
5. Return verdict + audit trail

### Research Inspiration

`llm-jury` is inspired by the CEJ (Collaborative Expert Judgment) module described in [arXiv:2512.23732](https://arxiv.org/abs/2512.23732). This package generalizes that pattern into a domain-agnostic SDK with pluggable classifiers, multiple debate modes, multiple judge strategies, threshold calibration, and Python + TypeScript distributions.

## Install

```bash
pip install llm-jury
```

Optional extras:

```bash
pip install "llm-jury[sklearn]"
pip install "llm-jury[huggingface]"
pip install "llm-jury[all]"
```

## Prerequisites

- Python `>=3.10`
- For real LLM calls: `OPENAI_API_KEY` (or provider key through your LiteLLM/OpenAI setup)

## Quick Start

```python
import asyncio
from llm_jury import Jury, PersonaRegistry
from llm_jury.classifiers import FunctionClassifier
from llm_jury.judges import MajorityVoteJudge

classifier = FunctionClassifier(
    fn=lambda text: ("safe", 0.62),
    labels=["safe", "unsafe"],
)

jury = Jury(
    classifier=classifier,
    personas=PersonaRegistry.content_moderation(),
    confidence_threshold=0.7,
    judge=MajorityVoteJudge(),
)

async def main():
    verdict = await jury.classify("borderline message")
    print(verdict.label, verdict.confidence, verdict.was_escalated)

asyncio.run(main())
```

### With LLM Classifier

```python
import asyncio
from llm_jury import Jury, PersonaRegistry
from llm_jury.classifiers import LLMClassifier
from llm_jury.judges import MajorityVoteJudge

classifier = LLMClassifier(labels=["safe", "unsafe"])

jury = Jury(
    classifier=classifier,
    personas=PersonaRegistry.content_moderation(),
    confidence_threshold=0.85,
    judge=MajorityVoteJudge(),
)

async def main():
    verdict = await jury.classify("That group always causes problems")
    print(f"Label: {verdict.label}")
    print(f"Confidence: {verdict.confidence}")
    print(f"Escalated: {verdict.was_escalated}")

    if verdict.debate_transcript:
        for resp in verdict.debate_transcript.rounds[-1]:
            print(f"  {resp.persona_name}: {resp.label} ({resp.confidence})")

asyncio.run(main())
```

## SDK Response

`jury.classify(text)` returns a `Verdict`. There are two shapes depending on whether the input was escalated.

### Fast path (confidence above threshold)

When the primary classifier is confident enough, the verdict is returned directly with no debate.

```json
{
  "label": "safe",
  "confidence": 0.95,
  "reasoning": "Classified by primary classifier with sufficient confidence.",
  "was_escalated": false,
  "primary_result": {
    "label": "safe",
    "confidence": 0.95,
    "raw_output": { "label": "safe", "confidence": 0.95 }
  },
  "debate_transcript": null,
  "judge_strategy": "primary_classifier",
  "total_duration_ms": 312,
  "total_cost_usd": 0.0001
}
```

### Escalated (confidence below threshold)

When confidence is too low, the input goes through persona debate and a judge produces the final verdict.

```json
{
  "label": "unsafe",
  "confidence": 1.0,
  "reasoning": "The statement is a sweeping negative generalization about an entire group of people.",
  "was_escalated": true,
  "primary_result": {
    "label": "unsafe",
    "confidence": 0.62,
    "raw_output": { "label": "unsafe", "confidence": 0.62 }
  },
  "debate_transcript": {
    "input_text": "Those people always cause problems wherever they go",
    "primary_result": { "label": "unsafe", "confidence": 0.62 },
    "rounds": [
      [
        {
          "persona_name": "Policy Analyst",
          "label": "unsafe",
          "confidence": 0.90,
          "reasoning": "The statement is a blanket negative generalization targeting a group.",
          "key_factors": ["group-targeting language", "sweeping generalization"],
          "dissent_notes": null,
          "tokens_used": 185,
          "cost_usd": 0.0003
        },
        {
          "persona_name": "Cultural Context Expert",
          "label": "unsafe",
          "confidence": 0.85,
          "reasoning": "While context could soften interpretation, the phrasing is unambiguously negative.",
          "key_factors": ["no mitigating context", "derogatory framing"],
          "dissent_notes": null,
          "tokens_used": 192,
          "cost_usd": 0.0003
        },
        {
          "persona_name": "Harm Assessment Specialist",
          "label": "unsafe",
          "confidence": 0.92,
          "reasoning": "Broad negative generalization risks normalizing prejudice against the targeted group.",
          "key_factors": ["potential for real-world harm", "targets unspecified group"],
          "dissent_notes": null,
          "tokens_used": 178,
          "cost_usd": 0.0003
        }
      ]
    ],
    "summary": "The experts unanimously agreed the statement constitutes an unsafe sweeping generalization targeting a group.",
    "duration_ms": 2450,
    "total_tokens": 555,
    "total_cost_usd": 0.0009
  },
  "judge_strategy": "majority_vote",
  "total_duration_ms": 2780,
  "total_cost_usd": 0.001
}
```

### Verdict field reference

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Final classification |
| `confidence` | `float` | Final confidence (0.0-1.0) |
| `reasoning` | `str` | Human-readable explanation |
| `was_escalated` | `bool` | Whether debate was triggered |
| `primary_result` | `ClassificationResult` | Fast-path classifier output |
| `debate_transcript` | `DebateTranscript \| None` | Full debate audit trail incl. `rounds`, `summary`, token/cost totals (null if not escalated) |
| `judge_strategy` | `str` | Strategy that produced the verdict |
| `total_duration_ms` | `int` | Wall-clock time (ms) |
| `total_cost_usd` | `float \| None` | API cost in USD |

### Persona response fields

| Field | Type | Description |
|---|---|---|
| `persona_name` | `str` | Which persona |
| `label` | `str` | This persona's classification |
| `confidence` | `float` | This persona's confidence |
| `reasoning` | `str` | Full reasoning chain |
| `key_factors` | `list[str]` | Key decision factors |
| `dissent_notes` | `str \| None` | Rebuttal in deliberation/adversarial modes |
| `tokens_used` | `int` | Tokens consumed |
| `cost_usd` | `float \| None` | API cost for this call |

`DebateTranscript` also includes `summary` (`str | None`) — a structured summary produced during the Summarisation stage of the deliberation pipeline (null in non-deliberation modes).

## Choosing What To Use

### Classifiers

| Classifier | When to use | Example |
|---|---|---|
| `FunctionClassifier` | Wrap an existing model or function | `FunctionClassifier(fn=my_model, labels=["a","b"])` |
| `LLMClassifier` | Primary classifier is an LLM | `LLMClassifier(labels=["safe","unsafe"])` |
| `HuggingFaceClassifier` | Local HuggingFace model | `HuggingFaceClassifier("unitary/toxic-bert")` |
| `SklearnClassifier` | Wrap a scikit-learn model | `SklearnClassifier(model, labels, vectorizer)` |

### Built-in Persona Sets

| Method | Domain | Personas |
|---|---|---|
| `PersonaRegistry.content_moderation()` | Trust & Safety | Policy Analyst, Cultural Context Expert, Harm Assessment Specialist |
| `PersonaRegistry.legal_compliance()` | Legal/Regulatory | Regulatory Attorney, Business Risk Analyst, Industry Standards Expert |
| `PersonaRegistry.medical_triage()` | Healthcare | Clinical Safety Reviewer, Contextual Historian, Resource Allocation Analyst |
| `PersonaRegistry.financial_compliance()` | AML/KYC | AML Investigator, Risk Quant, Business Controls Reviewer |
| `PersonaRegistry.custom([...])` | Any domain | Provide your own persona dicts |

### Judge Strategies

| Strategy | How it decides | Best for |
|---|---|---|
| `MajorityVoteJudge()` | Counts persona votes. Confidence = fraction agreeing. | Fast, no extra LLM call |
| `WeightedVoteJudge()` | Weights votes by persona confidence. | When confidence scores vary significantly |
| `LLMJudge()` | LLM reads full transcript and synthesises verdict. | Maximum quality, auditable reasoning |
| `BayesianJudge()` | Bayesian aggregation with optional persona priors. | When you have reliability data on personas |

### Debate Modes

| Mode | Behaviour | Best for |
|---|---|---|
| `independent` | All personas assess in parallel | Fast, low cost |
| `sequential` | Each persona sees previous responses | Building on earlier assessments |
| `deliberation` (default) | Full 4-stage CEJ pipeline: Initial Opinions, Structured Debate, Summarisation, Final Judgment | Maximum value; complex edge cases |
| `adversarial` | Assigns prosecution/defense stances | Stress-testing a classification |

## Important Notes

- **Temperature is handled automatically.** The SDK omits the temperature parameter for reasoning models (`gpt-5*`, `o1*`, `o3*`). No configuration needed.
- **Escalation is strictly `< threshold`** — confidence exactly equal to the threshold does NOT escalate.
- **Automatic retry**: All LLM calls retry up to 3 times with exponential backoff (via tenacity).
- **Default debate mode is deliberation** for maximum value — it runs the full 4-stage CEJ pipeline. For cheaper/faster operation, use `DebateConfig(mode=DebateMode.INDEPENDENT)`.
- **Cost tracking** — `total_cost_usd` on verdicts is estimated from token usage via litellm's model pricing table, not from provider billing. Accurate for known models; may be `None` for unrecognised ones.
- **Empty personas disables escalation**: If you pass `personas=[]`, the jury always returns the primary classifier result.

## API Reference

### Public Exports

```python
from llm_jury import (
    Jury,
    JuryStats,
    Persona,
    PersonaResponse,
    PersonaRegistry,
    DebateConfig,
    DebateMode,
    DebateTranscript,
    Verdict,
)
```

### `Jury` Options

| Option | Default | Description |
|---|---|---|
| `classifier` | (required) | Primary classifier |
| `personas` | (required) | List of personas |
| `confidence_threshold` | `0.7` | Escalation threshold |
| `judge` | `None` (defaults to `LLMJudge`) | Judge strategy |
| `debate_config` | `None` | Debate configuration |
| `escalation_override` | `None` | Force escalation |
| `max_debate_cost_usd` | `None` | Cost cap for debate |
| `debate_concurrency` | `5` | Max concurrent persona calls |
| `on_escalation` | `None` | Escalation callback |
| `on_verdict` | `None` | Verdict callback |
| `llm_client` | `None` | LLM transport override |
| `logger` | `None` | Logger override |

Methods:

- `await classify(text)` — classify a single input
- `await classify_batch(texts, concurrency=10)` — classify multiple inputs

Behavior notes:

- Escalation condition is strictly `< threshold` (exactly equal does not escalate).
- If `personas` is empty, jury escalation is effectively disabled.
- If `max_debate_cost_usd` is exceeded, result falls back to primary classifier with `judge_strategy` set to `cost_guard_primary_fallback`.

Stats: `jury.stats.total`, `fast_path`, `escalated`, `escalation_rate`, `cost_savings_vs_always_escalate`.

### `DebateConfig` Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `deliberation` | Debate mode |
| `max_rounds` | `2` | Max deliberation rounds |
| `include_primary_result` | `true` | Include primary result in prompts |
| `include_confidence` | `true` | Include confidence in prompt context |

### Personas

Persona fields: `name`, `role`, `system_prompt`, `model="gpt-5-mini"`, `temperature=0.3`, `known_bias=None`.

### Classifiers (API)

All classifiers implement `classify(text)` and expose `labels`.

- **FunctionClassifier**: `FunctionClassifier(fn, labels)` where `fn` may be sync or async
- **LLMClassifier**: `LLMClassifier(model="gpt-5-mini", labels=None, system_prompt=None, llm_client=None, temperature=0.0)`
- **SklearnClassifier**: `SklearnClassifier(model, labels, vectorizer=None)` uses `predict_proba`
- **HuggingFaceClassifier**: `HuggingFaceClassifier(model_name, device="cpu")` (requires `transformers`)

### Judge Strategies (API)

- **MajorityVoteJudge**: `MajorityVoteJudge()` — confidence = fraction of personas voting winning label
- **WeightedVoteJudge**: `WeightedVoteJudge()` — confidence based on confidence-weighted label scores
- **LLMJudge**: `LLMJudge(model="gpt-5-mini", system_prompt=None, temperature=0.0, llm_client=None)` — falls back to primary result with `llm_judge_fallback_invalid_json` if JSON parse fails
- **BayesianJudge**: `BayesianJudge(persona_priors=None)` — uses persona priors/reliability maps if provided

### Threshold Calibration

`ThresholdCalibrator(jury)` then `await calibrate(texts, labels, error_cost=10.0, escalation_cost=0.05, thresholds=None)`.

Report: `calibration_report()` returns rows with threshold, accuracy, escalation rate, and total cost. `calibrate(...)` mutates `jury.threshold` to the best threshold.

### LLM Transport (`LiteLLMClient`)

- `LiteLLMClient.complete(model, system_prompt, prompt, temperature)`
- Uses `litellm.acompletion`
- Returns: `{content, tokens, cost_usd}` (`cost_usd` may be `None`)
- Raises a runtime error if `litellm` is not installed and no custom `llm_client` is injected.

## Examples

Runnable examples in `examples/` (require `OPENAI_API_KEY`):

```bash
python examples/content_moderation.py   # Content moderation with LLM classifier
python examples/custom_personas.py      # Custom persona definitions + deliberation mode
python examples/legal_compliance.py     # Legal compliance with sequential debate + weighted vote
python examples/threshold_calibration.py # Threshold calibration (no API key needed)
```

## Testing

```bash
cd packages/python
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## CLI

The CLI is for batch workflows. The primary interface is the Python API above.

```bash
# Classify a JSONL file
llm-jury classify \
  --input input.jsonl \
  --output verdicts.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --threshold 0.7 \
  --labels safe,unsafe

# Calibrate threshold from labelled data
llm-jury calibrate \
  --input calibration.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --labels safe,unsafe
```

Input JSONL format for `classify`:
```json
{"text": "some text", "predicted_label": "safe", "predicted_confidence": 0.95}
```

Input JSONL format for `calibrate` (requires ground-truth `label`):
```json
{"text": "some text", "label": "safe", "predicted_label": "safe", "predicted_confidence": 0.95}
```

Supported classifier specs: `function`, `llm:<model>`, `huggingface:<model>`.

## License

MIT
