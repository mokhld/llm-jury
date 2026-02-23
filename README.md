# llm-jury

**When your classifier is uncertain, let a configurable jury of LLM personas debate and return an auditable verdict.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Node.js 22.6+](https://img.shields.io/badge/node.js-22.6%2B-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python Tests](https://img.shields.io/badge/python_tests-55%20passed-brightgreen.svg)]()
[![TypeScript Tests](https://img.shields.io/badge/ts_tests-38%20passed-brightgreen.svg)]()

## Overview

`llm-jury` is an SDK, not a hosted API. Your app imports it directly:

- Python package: `llm-jury` (import path: `llm_jury`)
- TypeScript package: `@llm-jury/core`

It wraps a classifier returning `(label, confidence)` and adds confidence-based escalation:

1. Run primary classifier (fast path)
2. Return directly when confidence is high
3. Escalate low-confidence cases to persona debate
4. Consolidate with a judge strategy
5. Return verdict + audit trail

## Research Inspiration

`llm-jury` is inspired by the CEJ (Collaborative Expert Judgment) module described in arXiv:2512.23732:

- https://arxiv.org/abs/2512.23732

This package generalizes that pattern into a domain-agnostic SDK with pluggable classifiers, multiple debate modes, multiple judge strategies, threshold calibration, and Python + TypeScript distributions.

## Install

### Python

```bash
pip install llm-jury
```

Optional extras:

```bash
pip install "llm-jury[sklearn]"
pip install "llm-jury[huggingface]"
pip install "llm-jury[all]"
```

### TypeScript

```bash
npm install @llm-jury/core
```

## Prerequisites

- Python `>=3.10`
- Node.js `>=22.6`
- For real LLM calls: `OPENAI_API_KEY` (or provider key through your LiteLLM/OpenAI setup)

## Quick Start

### Python (copy-paste runnable)

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

### Python (with LLM classifier)

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

### TypeScript

```ts
import {
  FunctionClassifier,
  Jury,
  MajorityVoteJudge,
  PersonaRegistry,
} from "@llm-jury/core";

const classifier = new FunctionClassifier(
  () => ["safe", 0.62],
  ["safe", "unsafe"],
);

const jury = new Jury({
  classifier,
  personas: PersonaRegistry.contentModeration(),
  confidenceThreshold: 0.7,
  judge: new MajorityVoteJudge(),
});

const verdict = await jury.classify("borderline message");
console.log(verdict.label, verdict.confidence, verdict.wasEscalated);
```

## Examples

Runnable examples in `examples/` (require `OPENAI_API_KEY`):

```bash
python examples/content_moderation.py   # Content moderation with LLM classifier
python examples/custom_personas.py      # Custom persona definitions + deliberation mode
python examples/legal_compliance.py     # Legal compliance with sequential debate + weighted vote
python examples/threshold_calibration.py # Threshold calibration (no API key needed)
```

## SDK Response

`jury.classify(text)` returns a `Verdict`. There are two shapes depending on whether the input was escalated.

### Fast path (confidence above threshold)

When the primary classifier is confident enough, the verdict is returned directly with no debate.

**Python:**

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

**TypeScript:**

```json
{
  "label": "safe",
  "confidence": 0.95,
  "reasoning": "Classified by primary classifier with sufficient confidence.",
  "wasEscalated": false,
  "primaryResult": {
    "label": "safe",
    "confidence": 0.95,
    "rawOutput": { "label": "safe", "confidence": 0.95 }
  },
  "debateTranscript": null,
  "judgeStrategy": "primary_classifier",
  "totalDurationMs": 312,
  "totalCostUsd": 0.0001
}
```

### Escalated (confidence below threshold)

When confidence is too low, the input goes through persona debate and a judge produces the final verdict.

**Python:**

```json
{
  "label": "unsafe",
  "confidence": 1.0,
  "reasoning": "The statement is a sweeping negative generalization about an entire group of people, attributing harmful behavior broadly.",
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

**TypeScript:**

```json
{
  "label": "unsafe",
  "confidence": 1.0,
  "reasoning": "The statement is a sweeping negative generalization about an entire group of people.",
  "wasEscalated": true,
  "primaryResult": {
    "label": "unsafe",
    "confidence": 0.62,
    "rawOutput": { "label": "unsafe", "confidence": 0.62 }
  },
  "debateTranscript": {
    "inputText": "Those people always cause problems wherever they go",
    "primaryResult": { "label": "unsafe", "confidence": 0.62 },
    "rounds": [
      [
        {
          "personaName": "Policy Analyst",
          "label": "unsafe",
          "confidence": 0.90,
          "reasoning": "The statement is a blanket negative generalization targeting a group.",
          "keyFactors": ["group-targeting language", "sweeping generalization"],
          "dissentNotes": null,
          "tokensUsed": 185,
          "costUsd": 0.0003
        },
        {
          "personaName": "Cultural Context Expert",
          "label": "unsafe",
          "confidence": 0.85,
          "reasoning": "While context could soften interpretation, the phrasing is unambiguously negative.",
          "keyFactors": ["no mitigating context", "derogatory framing"],
          "dissentNotes": null,
          "tokensUsed": 192,
          "costUsd": 0.0003
        },
        {
          "personaName": "Harm Assessment Specialist",
          "label": "unsafe",
          "confidence": 0.92,
          "reasoning": "Broad negative generalization risks normalizing prejudice against the targeted group.",
          "keyFactors": ["potential for real-world harm", "targets unspecified group"],
          "dissentNotes": null,
          "tokensUsed": 178,
          "costUsd": 0.0003
        }
      ]
    ],
    "summary": "The experts unanimously agreed the statement constitutes an unsafe sweeping generalization targeting a group.",
    "durationMs": 2450,
    "totalTokens": 555,
    "totalCostUsd": 0.0009
  },
  "judgeStrategy": "majority_vote",
  "totalDurationMs": 2780,
  "totalCostUsd": 0.001
}
```

### Verdict field reference

| Python | TypeScript | Type | Description |
|---|---|---|---|
| `label` | `label` | `str` / `string` | Final classification |
| `confidence` | `confidence` | `float` / `number` | Final confidence (0.0–1.0) |
| `reasoning` | `reasoning` | `str` / `string` | Human-readable explanation |
| `was_escalated` | `wasEscalated` | `bool` / `boolean` | Whether debate was triggered |
| `primary_result` | `primaryResult` | `ClassificationResult` | Fast-path classifier output |
| `debate_transcript` | `debateTranscript` | `DebateTranscript \| None` | Full debate audit trail incl. `rounds`, `summary`, token/cost totals (null if not escalated) |
| `judge_strategy` | `judgeStrategy` | `str` / `string` | Strategy that produced the verdict |
| `total_duration_ms` | `totalDurationMs` | `int` / `number` | Wall-clock time (ms) |
| `total_cost_usd` | `totalCostUsd` | `float \| None` / `number \| null` | API cost in USD |

### Persona response fields

| Python | TypeScript | Type | Description |
|---|---|---|---|
| `persona_name` | `personaName` | `str` / `string` | Which persona |
| `label` | `label` | `str` / `string` | This persona's classification |
| `confidence` | `confidence` | `float` / `number` | This persona's confidence |
| `reasoning` | `reasoning` | `str` / `string` | Full reasoning chain |
| `key_factors` | `keyFactors` | `list[str]` / `string[]` | Key decision factors |
| `dissent_notes` | `dissentNotes` | `str \| None` / `string \| null` | Rebuttal in deliberation/adversarial modes |
| `tokens_used` | `tokensUsed` | `int` / `number` | Tokens consumed |
| `cost_usd` | `costUsd` | `float \| None` / `number \| null` | API cost for this call |

`DebateTranscript` also includes `summary` (Python: `str | None`, TypeScript: `string?`) — a structured summary produced during the Summarisation stage of the deliberation pipeline (null/undefined in non-deliberation modes).

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
| `deliberation` (default) | Full 4-stage CEJ pipeline: Initial Opinions → Structured Debate → Summarisation → Final Judgment | Maximum value; complex edge cases |
| `adversarial` | Assigns prosecution/defense stances | Stress-testing a classification |

### Important Notes

- **Temperature is handled automatically.** The SDK omits the temperature parameter for reasoning models (`gpt-5*`, `o1*`, `o3*`). No configuration needed.
- **Escalation is strictly `< threshold`** — confidence exactly equal to the threshold does NOT escalate.
- **Automatic retry**: All LLM calls retry up to 3 times with exponential backoff (via tenacity).
- **Default debate mode is deliberation** for maximum value — it runs the full 4-stage CEJ pipeline. For cheaper/faster operation, use `DebateConfig(mode=DebateMode.INDEPENDENT)` (Python) or `{ mode: DebateMode.Independent }` (TypeScript).
- **Cost tracking is Python-only** — `total_cost_usd` on verdicts is estimated from token usage via litellm's model pricing table, not from provider billing. Accurate for known models; may be `None` for unrecognised ones. In TypeScript, `totalCostUsd` is always `undefined` unless a custom `llmClient` provides cost data (no viable npm cost-estimation library exists).
- **Empty personas disables escalation**: If you pass `personas=[]`, the jury always returns the primary classifier result.

## SDK API Reference

### Public Exports

### Python

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

### TypeScript

```ts
import {
  Jury,
  JuryStats,
  DebateConfig,
  DebateMode,
  PersonaRegistry,
  FunctionClassifier,
  LLMClassifier,
  HuggingFaceClassifier,
  SklearnClassifier,
  MajorityVoteJudge,
  WeightedVoteJudge,
  LLMJudge,
  BayesianJudge,
  ThresholdCalibrator,
  LiteLLMClient,
} from "@llm-jury/core";
```

### `Jury` Options

| Concept | Python (`Jury(...)`) | TypeScript (`new Jury({...})`) |
|---|---|---|
| Primary classifier | `classifier` | `classifier` |
| Personas | `personas` | `personas` |
| Confidence threshold | `confidence_threshold=0.7` | `confidenceThreshold=0.7` |
| Judge strategy | `judge=None` (defaults to `LLMJudge`) | `judge` (defaults to `LLMJudge`) |
| Debate config | `debate_config=None` | `debateConfig` |
| Escalation override | `escalation_override=None` | `escalationOverride` |
| Debate cost cap | `max_debate_cost_usd=None` | `maxDebateCostUsd` |
| Debate concurrency | `debate_concurrency=5` | `debateConcurrency=5` |
| Escalation callback | `on_escalation=None` | `onEscalation` |
| Verdict callback | `on_verdict=None` | `onVerdict` |
| LLM transport override | `llm_client=None` | `llmClient` |
| Logger override | `logger=None` | n/a |

Methods:

- Python: `await classify(text)`, `await classify_batch(texts, concurrency=10)`
- TypeScript: `await classify(text)`, `await classifyBatch(texts, concurrency=10)`

Behavior notes:

- Escalation condition is strictly `< threshold` (exactly equal does not escalate).
- If `personas` is empty, jury escalation is effectively disabled.
- If `max_debate_cost_usd` / `maxDebateCostUsd` is exceeded, result falls back to primary classifier with `judge_strategy` / `judgeStrategy` set to `cost_guard_primary_fallback`.

Stats:

- Python: `jury.stats.total`, `fast_path`, `escalated`, `escalation_rate`, `cost_savings_vs_always_escalate`
- TypeScript: `jury.stats.total`, `fastPath`, `escalated`, `escalationRate`, `costSavingsVsAlwaysEscalate`

### `DebateConfig` Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `deliberation` | Debate mode |
| `max_rounds` / `maxRounds` | `2` | Max deliberation rounds |
| `include_primary_result` / `includePrimaryResult` | `true` | Include primary result in prompts |
| `include_confidence` / `includeConfidence` | `true` | Include confidence in prompt context |

Modes:

- `deliberation` (default): full 4-stage CEJ pipeline with optional early consensus
- `independent`: all personas respond independently
- `sequential`: personas see previous responses in order
- `adversarial`: assigns prosecution/defense stances

### Personas

Persona fields:

- Python: `name`, `role`, `system_prompt`, `model="gpt-5-mini"`, `temperature=0.3`, `known_bias=None`
- TypeScript: `name`, `role`, `systemPrompt`, `model="gpt-5-mini"`, `temperature=0.3`, `knownBias?`

Built-in registries:

- Python: `PersonaRegistry.content_moderation()`, `legal_compliance()`, `medical_triage()`, `financial_compliance()`, `custom(...)`
- TypeScript: `PersonaRegistry.contentModeration()`, `legalCompliance()`, `medicalTriage()`, `financialCompliance()`, `custom(...)`

### Classifiers

All classifiers implement `classify(text)` and expose `labels`.

#### Function Classifier

- Python: `FunctionClassifier(fn, labels)` where `fn` may be sync or async
- TypeScript: `new FunctionClassifier(fn, labels)` where `fn` may return tuple or Promise tuple

#### LLM Classifier

- Python: `LLMClassifier(model="gpt-5-mini", labels=None, system_prompt=None, llm_client=None, temperature=0.0)`
- TypeScript: `new LLMClassifier({ model, labels, systemPrompt, llmClient, temperature })`

Behavior notes:

- If `system_prompt` / `systemPrompt` is not set, a default classification prompt is applied automatically.
- Expects model JSON response with `label` and `confidence`
- If JSON parse fails, falls back to first label (or `"unknown"`) with `confidence=0`

#### Sklearn Classifier

- Python: `SklearnClassifier(model, labels, vectorizer=None)` uses `predict_proba`
- TypeScript: `new SklearnClassifier(model, labels, vectorizer?)` where model has `predictProba(...)`

#### HuggingFace Classifier

- Python: `HuggingFaceClassifier(model_name, device="cpu")` (requires `transformers`)
- TypeScript: `new HuggingFaceClassifier({ modelName?, device?, pipeline? })`
  - Uses injected `pipeline` or loads `@xenova/transformers`
  - Must provide `modelName` or `pipeline` (constructor throws otherwise)

### Judge Strategies

#### Majority Vote

- Python: `MajorityVoteJudge()`
- TypeScript: `new MajorityVoteJudge()`

Final confidence = fraction of personas voting winning label.

#### Weighted Vote

- Python: `WeightedVoteJudge()`
- TypeScript: `new WeightedVoteJudge()`

Final confidence based on confidence-weighted label scores.

#### LLM Judge

- Python: `LLMJudge(model="gpt-5-mini", system_prompt=None, temperature=0.0, llm_client=None)`
- TypeScript: `new LLMJudge({ model, systemPrompt, temperature, llmClient })`

Behavior notes:

- Judge receives full transcript prompt
- If JSON parse fails, falls back to primary result with strategy marker:
  - `llm_judge_fallback_invalid_json`

#### Bayesian Judge

- Python: `BayesianJudge(persona_priors=None)`
- TypeScript: `new BayesianJudge(priors={})`

Uses persona priors/reliability maps if provided.

### Threshold Calibration

- Python: `ThresholdCalibrator(jury)` then `await calibrate(texts, labels, error_cost=10.0, escalation_cost=0.05, thresholds=None)`
- TypeScript: `new ThresholdCalibrator(jury)` then `await calibrate({ texts, labels, errorCost=10, escalationCost=0.05, thresholds? })`

Report:

- Python: `calibration_report()`
- TypeScript: `calibrationReport()`

Both return rows with threshold, accuracy, escalation rate, and total cost.
`calibrate(...)` mutates `jury.threshold` to the best threshold.

Default threshold candidates when not provided:

- `0.50, 0.55, ..., 0.95`

### LLM Transport (`LiteLLMClient`)

#### Python

- `LiteLLMClient.complete(model, system_prompt, prompt, temperature)`
- Uses `litellm.acompletion`
- Returns: `{content, tokens, cost_usd}` (`cost_usd` may be `None`)
- Raises a runtime error if `litellm` is not installed and no custom `llm_client` is injected.

#### TypeScript

- `new LiteLLMClient({ baseUrl?, apiKey?, timeoutMs? })`
- Falls back to env vars:
  - `LITELLM_BASE_URL`, `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
  - `LITELLM_API_KEY`, `OPENAI_API_KEY`
- Sends `POST /chat/completions`
- Returns `{content, tokens, costUsd}` — `costUsd` is always `undefined` because no viable npm cost-estimation library exists. A custom `llmClient` can provide cost data if needed.
- Throws before request if no API key is configured.

Temperature is automatically omitted for reasoning models (`gpt-5*`, `o1*`, `o3*`).

## Testing

### Python

```bash
cd packages/python
pip install -e ".[dev]"
python -m pytest tests/ -v
```

### TypeScript

```bash
cd packages/typescript
npm test
```

### Monorepo

```bash
npm test
```

## Releasing

Release automation is defined in `.github/workflows/release.yml`.

Trigger modes:

- `release.published`: publishes to PyPI and npm
- `workflow_dispatch`: lets you pick `python_repository` (`none`, `testpypi`, `pypi`) and whether to `publish_npm`

Requirements:

- Create GitHub environments: `testpypi`, `pypi`, `npm`
- Configure PyPI and TestPyPI trusted publishing for this repository/workflow
- Add `NPM_TOKEN` secret in environment `npm` (or repo secrets)
- Tag format must be `vX.Y.Z` and must match:
  - `packages/python/pyproject.toml` version
  - `packages/typescript/package.json` version

Typical flow:

1. Bump both package versions to the same value (for example `0.1.1`).
2. Create and push tag `v0.1.1`.
3. Publish:
   - GitHub Release `published` for production publish
   - `workflow_dispatch` for TestPyPI dry runs and selective publish targets

## CLI (Secondary)

The product is SDK-first. CLI is provided for batch workflows.

### Commands

- `llm-jury classify`
- `llm-jury calibrate`

### Common CLI options

- `--classifier` (`function`, `llm:<model>`, `huggingface:<model>`) default `function`
- `--personas` (`content_moderation`, `legal_compliance`, `medical_triage`, `financial_compliance`)
- `--labels` comma-separated labels
- `--judge` (`llm`, `majority`, `weighted`, `bayesian`)
- `--judge-model` default `gpt-5-mini`
- `--persona-model` default `gpt-5-mini`
- `--debate-mode` (`independent`, `sequential`, `deliberation`, `adversarial`)
- `--max-rounds` default `1`
- `--max-debate-cost`
- `--debate-concurrency` default `5`
- `--hide-primary-result`
- `--hide-confidence`

### Classify-only options

- `--input` (required)
- `--output` (required)
- `--threshold` default `0.7`
- `--concurrency` default `10`

### Calibrate-only options

- `--input` (required, must include ground-truth `label` per row)
- `--error-cost` default `10.0`
- `--escalation-cost` default `0.05`
- `--initial-threshold` default `0.7`

## License

MIT
