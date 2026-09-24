# llm-jury

**When your classifier is uncertain, let a configurable jury of LLM personas debate and return an auditable verdict.**

[![CI](https://github.com/mokhld/llm-jury/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mokhld/llm-jury/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Node.js 20+](https://img.shields.io/badge/node.js-20%2B-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

`llm-jury` is an SDK, not a hosted API. Your app imports it directly:

- Python package: `llm-jury-classifier` (import path: `llm_jury`)
- TypeScript package: `@llm-jury/core`

Do not `pip install llm-jury`: that name on PyPI is an unrelated project that also installs an `llm_jury` module.

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
pip install llm-jury-classifier
```

Optional extras:

```bash
pip install "llm-jury-classifier[sklearn]"
pip install "llm-jury-classifier[huggingface]"
pip install "llm-jury-classifier[all]"
```

### TypeScript

```bash
npm install @llm-jury/core
```

## Prerequisites

- Python `>=3.10`
- Node.js `>=20` for the published package; `>=22.6` to run the TypeScript examples or the repo's tests, which load `.ts` files with `--experimental-strip-types`
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

Runnable examples in `examples/` (require `OPENAI_API_KEY` except the calibration and evaluation ones):

**Python** (`examples/*.py`):

```bash
python examples/content_moderation.py    # Content moderation with LLM classifier
python examples/custom_personas.py       # Custom persona definitions + deliberation mode
python examples/legal_compliance.py      # Legal compliance with sequential debate + weighted vote
python examples/threshold_calibration.py # Threshold calibration (no API key needed)
python examples/evaluate_jury.py         # Measure the jury against labelled data (no API key needed)
```

**TypeScript** (`examples/typescript/*.ts`):

```bash
node --experimental-strip-types examples/typescript/content_moderation.ts
node --experimental-strip-types examples/typescript/custom_personas.ts
node --experimental-strip-types examples/typescript/legal_compliance.ts
node --experimental-strip-types examples/typescript/threshold_calibration.ts
node --experimental-strip-types examples/typescript/evaluate_jury.ts
```

TS examples import from `@llm-jury/core`; to run from a fresh clone, install the local package first (`cd packages/typescript && npm install && npm link`, then `npm link @llm-jury/core` from the repo root) or copy the example into a project where the package is already installed.

## SDK Response

`jury.classify(text)` returns a `Verdict`. There are two shapes depending on whether the input was escalated. The JSON below is what `to_dict()` / `toDict()` return. The samples are illustrative and shortened (persona `raw_response` is left out). The durations and costs are made up; real escalations take far longer and cost more (see [Important Notes](#important-notes) for measured numbers).

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
    "raw_output": { "label": "safe", "confidence": 0.95 },
    "cost_usd": 0.0001
  },
  "debate_transcript": null,
  "judge_strategy": "primary_classifier",
  "total_duration_ms": 312,
  "total_cost_usd": 0.0001,
  "persona_failures": 0,
  "library_version": "0.2.0",
  "created_at": "2026-09-24T10:15:00.123456+00:00",
  "judge_details": null,
  "debate_degraded": false
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
    "rawOutput": { "label": "safe", "confidence": 0.95 },
    "costUsd": 0.0001
  },
  "debateTranscript": null,
  "judgeStrategy": "primary_classifier",
  "totalDurationMs": 312,
  "totalCostUsd": 0.0001,
  "personaFailures": 0,
  "debateDegraded": false,
  "judgeDetails": null,
  "libraryVersion": "0.2.0",
  "createdAt": "2026-09-24T10:15:00.123Z"
}
```

### Escalated (confidence below threshold)

When confidence is too low, the input goes through persona debate and a judge produces the final verdict. In this sample the three personas agree in the opening round, so the debate stops there with no second round and no summary.

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
    "raw_output": { "label": "unsafe", "confidence": 0.62 },
    "cost_usd": 0.0001
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
          "cost_usd": 0.0003,
          "failed": false
        },
        {
          "persona_name": "Cultural Context Expert",
          "label": "unsafe",
          "confidence": 0.85,
          "reasoning": "While context could soften interpretation, the phrasing is unambiguously negative.",
          "key_factors": ["no mitigating context", "derogatory framing"],
          "dissent_notes": null,
          "tokens_used": 192,
          "cost_usd": 0.0003,
          "failed": false
        },
        {
          "persona_name": "Harm Assessment Specialist",
          "label": "unsafe",
          "confidence": 0.92,
          "reasoning": "Broad negative generalization risks normalizing prejudice against the targeted group.",
          "key_factors": ["potential for real-world harm", "targets unspecified group"],
          "dissent_notes": null,
          "tokens_used": 178,
          "cost_usd": 0.0003,
          "failed": false
        }
      ]
    ],
    "duration_ms": 2450,
    "total_tokens": 555,
    "total_cost_usd": 0.0009,
    "summary": null,
    "unpriced_calls": 0,
    "persona_biases": {
      "Policy Analyst": "policy-strict",
      "Cultural Context Expert": "tends permissive on context",
      "Harm Assessment Specialist": "harm-focused"
    }
  },
  "judge_strategy": "majority_vote",
  "total_duration_ms": 2780,
  "total_cost_usd": 0.001,
  "persona_failures": 0,
  "library_version": "0.2.0",
  "created_at": "2026-09-24T10:15:03.456789+00:00",
  "judge_details": null,
  "debate_degraded": false
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
    "rawOutput": { "label": "unsafe", "confidence": 0.62 },
    "costUsd": 0.0001
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
          "tokensUsed": 185,
          "costUsd": 0.0003,
          "failed": false
        },
        {
          "personaName": "Cultural Context Expert",
          "label": "unsafe",
          "confidence": 0.85,
          "reasoning": "While context could soften interpretation, the phrasing is unambiguously negative.",
          "keyFactors": ["no mitigating context", "derogatory framing"],
          "tokensUsed": 192,
          "costUsd": 0.0003,
          "failed": false
        },
        {
          "personaName": "Harm Assessment Specialist",
          "label": "unsafe",
          "confidence": 0.92,
          "reasoning": "Broad negative generalization risks normalizing prejudice against the targeted group.",
          "keyFactors": ["potential for real-world harm", "targets unspecified group"],
          "tokensUsed": 178,
          "costUsd": 0.0003,
          "failed": false
        }
      ]
    ],
    "durationMs": 2450,
    "totalTokens": 555,
    "totalCostUsd": 0.0009,
    "unpricedCalls": 0,
    "personaBiases": {
      "Policy Analyst": "policy-strict",
      "Cultural Context Expert": "tends permissive on context",
      "Harm Assessment Specialist": "harm-focused"
    }
  },
  "judgeStrategy": "majority_vote",
  "totalDurationMs": 2780,
  "totalCostUsd": 0.001,
  "personaFailures": 0,
  "debateDegraded": false,
  "judgeDetails": null,
  "libraryVersion": "0.2.0",
  "createdAt": "2026-09-24T10:15:03.456Z"
}
```

With the default TypeScript `LiteLLMClient`, LLM calls report no cost: persona `costUsd` is absent, `unpricedCalls` counts every call and `totalCostUsd` is `null` (see [Important Notes](#important-notes)).

### Verdict field reference

| Python | TypeScript | Type | Description |
|---|---|---|---|
| `label` | `label` | `str` / `string` | Final classification |
| `confidence` | `confidence` | `float` / `number` | Final confidence (0.0 to 1.0) |
| `reasoning` | `reasoning` | `str` / `string` | Human-readable explanation |
| `was_escalated` | `wasEscalated` | `bool` / `boolean` | Whether debate was triggered |
| `primary_result` | `primaryResult` | `ClassificationResult` | Fast-path classifier output |
| `debate_transcript` | `debateTranscript` | `DebateTranscript \| None` | Full debate audit trail (see below); null if no debate ran |
| `judge_strategy` | `judgeStrategy` | `str` / `string` | Strategy that produced the verdict, including the fallback markers listed under [Troubleshooting](#troubleshooting) |
| `total_duration_ms` | `totalDurationMs` | `int` / `number` | Wall-clock time (ms) |
| `total_cost_usd` | `totalCostUsd` | `float \| None` / `number \| null` | Primary classifier plus debate and judge cost in USD. `None` / `null` when any part is unknown; a lower bound when `debate_transcript.unpriced_calls > 0` |
| `persona_failures` | `personaFailures` | `int` / `number` | Persona calls across the debate that failed (LLM error, unparseable output, or a label outside the configured set) |
| `debate_degraded` | `debateDegraded` | `bool` / `boolean` | True when `persona_failures > 0`: the verdict was decided by fewer jurors than configured. Useful for routing to human review |
| `judge_details` | `judgeDetails` | `dict \| None` / `Record<string, unknown> \| null` | `LLMJudge` only: `key_agreements`, `key_disagreements`, `decisive_factor` (TS keys are camelCase). Null for other judges and fallbacks |
| `library_version` | `libraryVersion` | `str` / `string` | SDK version that produced the verdict |
| `created_at` | `createdAt` | `str` / `string` | ISO 8601 UTC timestamp |

### Debate transcript fields

| Python | TypeScript | Type | Description |
|---|---|---|---|
| `input_text` | `inputText` | `str` / `string` | The text that was classified |
| `primary_result` | `primaryResult` | `ClassificationResult` | Primary classifier output |
| `rounds` | `rounds` | list of persona response lists | One list per round, in order |
| `summary` | `summary` | `str \| None` / `string?` | Summariser output in deliberation mode. `None` / `undefined` when the debate stopped early, in other modes, or when the summariser call failed |
| `duration_ms` | `durationMs` | `int` / `number` | Debate wall-clock time (ms) |
| `total_tokens` | `totalTokens` | `int` / `number` | Tokens used by persona and summariser calls |
| `total_cost_usd` | `totalCostUsd` | `float \| None` / `number \| null` | Sum of the persona and summariser calls that reported a cost; null when none did |
| `unpriced_calls` | `unpricedCalls` | `int` / `number` | Debate calls that reported no cost. Non-zero means `total_cost_usd` is a lower bound |
| `persona_biases` | `personaBiases` | `dict[str, str]` / `Record<string, string>` | Persona name to `known_bias`, for personas that declare one. The LLM judge sees these |

### Persona response fields

| Python | TypeScript | Type | Description |
|---|---|---|---|
| `persona_name` | `personaName` | `str` / `string` | Which persona |
| `label` | `label` | `str` / `string` | This persona's classification |
| `confidence` | `confidence` | `float` / `number` | This persona's confidence |
| `reasoning` | `reasoning` | `str` / `string` | Full reasoning chain |
| `key_factors` | `keyFactors` | `list[str]` / `string[]` | Key decision factors |
| `dissent_notes` | `dissentNotes` | `str \| None` / `string?` | Rebuttal in deliberation/adversarial modes |
| `raw_response` | `rawResponse` | `str \| None` / `string?` | The model's raw reply |
| `tokens_used` | `tokensUsed` | `int` / `number` | Tokens consumed |
| `cost_usd` | `costUsd` | `float \| None` / `number?` | API cost for this call; null when the client reported none |
| `failed` | `failed` | `bool` / `boolean?` | True when this response is a placeholder for a failed persona call. Failed responses stay in the transcript for audit but carry no vote |

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

- **Temperature is handled automatically.** The SDK omits the temperature parameter for reasoning models (`gpt-5*`, `o1*`, `o3*`, also behind a provider prefix such as `openai/gpt-5-mini`). No configuration needed.
- **Escalation is strictly `< threshold`**: confidence exactly equal to the threshold does NOT escalate. A missing or non-numeric primary confidence (NaN, a string) always escalates.
- **Automatic retry**: each LLM call gets 3 attempts in total (the first try plus 2 retries) on connection errors, timeouts, 429 and 5xx responses; other errors fail at once. The client waits for the provider's `Retry-After` header when it sends one (capped at 60 s) and backs off exponentially otherwise. Change the count with `LiteLLMClient(max_attempts=...)` / `new LiteLLMClient({ maxAttempts })`.
- **Default debate mode is deliberation**, the full 4-stage CEJ pipeline. For cheaper and faster runs use `DebateConfig(mode=DebateMode.INDEPENDENT)` (Python) or `new DebateConfig({ mode: DebateMode.INDEPENDENT })` (TypeScript).
- **Deliberation stops early** after any round, the opening one included, when the personas' labels are unanimous or `early_stop_min_confidence` is met. A debate that stops early has no summary.
- **Latency and cost of an escalation**: a debate makes several rounds of LLM calls. Live runs in February 2026 with the default `gpt-5-mini`, three personas and a mix of debate modes and judges took 27 to 57 s and cost $0.007 to $0.015 per escalated item. Use that as a rough guide only; your models, personas and inputs will change it. The fast path costs one primary classifier call.
- **Cost tracking**: Python estimates each call's cost from litellm's model pricing table (not from provider billing), so a model litellm does not price reports `None`. The TypeScript `LiteLLMClient` never reports cost; inject a custom `llmClient` that returns `costUsd` if you need it. Unknown cost is `None` / `null`, never 0. An escalated `total_cost_usd` includes the primary classifier, is `None` / `null` when the primary cost or the whole debate cost is unknown, and is a lower bound when `debate_transcript.unpriced_calls > 0`.
- **The cost cap is checked twice.** Before a debate, `estimated_max_debate_cost_usd` is compared with `max_debate_cost_usd`; the estimate counts every persona call in every round, the summariser in deliberation mode and the judge when it is an `LLMJudge`, each at `estimated_cost_per_persona_usd` (default $0.01 per call). During the debate, reported spend is compared with the cap, and calls that reported no cost are charged at that same per-call estimate.
- **Empty personas disables escalation**: If you pass `personas=[]`, the jury always returns the primary classifier result.

## Prompt injection and untrusted input

The text you classify ends up inside the prompts of `LLMClassifier`, every persona, the summariser and `LLMJudge`. Escalated items are the borderline ones, which is also where a crafted input would aim.

What the SDK does:

- Every prompt puts the input inside `<input>` tags, after a note telling the model that the content is untrusted data and not instructions. Any `<input>` or `</input>` tags in the text are neutralised so it cannot close the fence early.
- `LLMClassifier`, `LLMJudge` and the personas request JSON output whose `label` is an enum of your labels. Providers that enforce strict `json_schema` output can only answer with those labels.
- Every label a model returns is checked against your labels (exact match, then case-insensitive). An out-of-set persona label counts as a failed response, and an out-of-set judge label falls back to a vote over the personas (`llm_judge_fallback_invalid_label`), so a label the input smuggles in never becomes the verdict.

What you should still do:

- Cap input length before calling `classify`. The SDK has no limit, and every call in a debate pays for the full input.
- Treat verdicts on adversarial input with care. Fencing makes injection harder, not impossible, and a persuasive input can still push the personas toward one of your labels.
- Route degraded and fallback verdicts to human review: `debate_degraded` / `debateDegraded` is true, or `judge_strategy` is `cost_guard_*` or `llm_judge_fallback_*`.
- Never execute, `eval` or template anything taken from a verdict's reasoning text.

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
    ThresholdCalibrator,
    JuryEvaluator,
    EvaluationReport,
    CachingLLMClient,
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
  JuryEvaluator,
  EvaluationReport,
  LiteLLMClient,
  CachingLLMClient,
} from "@llm-jury/core";
```

Wrap the LLM client with `CachingLLMClient` for an opt-in LRU
response cache keyed on `(model, system_prompt, prompt, temperature,
response_format)`. See per-package READMEs for usage.

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
| Estimated cost per LLM call | `estimated_cost_per_persona_usd=0.01` | `estimatedCostPerPersonaUsd=0.01` |
| Debate concurrency | `debate_concurrency=5` | `debateConcurrency=5` |
| Escalation callback | `on_escalation=None` | `onEscalation` |
| Cost-estimate gate | `on_cost_estimate=None` | `onCostEstimate` |
| Verdict callback (every verdict) | `on_verdict=None` | `onVerdict` |
| LLM transport override | `llm_client=None` | `llmClient` |
| Logger override | `logger=None` (defaults to `logging.getLogger(__name__)`) | `logger` (defaults to silent `NOOP_LOGGER`; pass `console` to opt in) |

Methods:

- Python: `await classify(text)`, `await classify_batch(texts, concurrency=10, return_exceptions=False)`
- TypeScript: `await classify(text)`, `await classifyBatch(texts, concurrency=10, returnExceptions=false)`

By default a failing text rejects the whole batch. Pass `return_exceptions=True` / `returnExceptions: true` to get the exception in that text's slot instead, so one bad row cannot discard the verdicts (and spend) of the rows that succeeded.

Behavior notes:

- Escalation condition is strictly `< threshold` (exactly equal does not escalate).
- If `personas` is empty, jury escalation is effectively disabled.
- `on_verdict` / `onVerdict` fires once for every verdict `classify` returns: fast path, cost-guard fallbacks and judged verdicts.
- `estimated_max_debate_cost_usd` / `estimatedMaxDebateCostUsd` is the pre-flight estimate: (persona calls + summariser call in deliberation mode + 1 for an `LLMJudge`) x `estimated_cost_per_persona_usd`. Persona calls are `len(personas) x max_rounds` in deliberation mode and `len(personas)` in the other modes. If it exceeds the cap, no debate runs and `judge_strategy` is `cost_guard_pre_flight`.
- If spend during the debate exceeds `max_debate_cost_usd` / `maxDebateCostUsd`, the result falls back to the primary classifier with `judge_strategy` / `judgeStrategy` set to `cost_guard_primary_fallback`.

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
| `early_stop_min_confidence` / `earlyStopMinConfidence` | `None` / `undefined` | Opt-in early stop for deliberation mode. When set, the debate ends after any round, the opening one included, whose **lowest** persona confidence is `>=` this value, even if personas disagree on label. Unanimous labels end it regardless. |

Modes:

- `deliberation` (default): full 4-stage CEJ pipeline; ends after any round whose labels are unanimous (no summary then)
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
- Sends a JSON schema with your labels as an enum and expects `label` and `confidence` back. The label is matched to your labels (exact, then case-insensitive) and returned in your spelling.
- Output it cannot use returns confidence 0, so the jury escalates it, with the reason in `raw_output["error"]` / `rawOutput.error`: `invalid_json` or `label_not_in_labels` (label becomes the first configured label), or `invalid_confidence` (label kept).

#### Sklearn Classifier

- Python: `SklearnClassifier(model, labels, vectorizer=None)` uses `predict_proba`
- TypeScript: `new SklearnClassifier(model, labels, vectorizer?)` where model has `predictProba(...)`
- Probability columns are named by the model's `classes_` (TS: `classes`) when those are the same set as `labels`, otherwise by position. A label count that differs from `classes_` raises. Python runs inference in a worker thread.

#### HuggingFace Classifier

- Python: `HuggingFaceClassifier(model_name, device="cpu", labels=None)` (requires `transformers`; inference runs in a worker thread)
- TypeScript: `new HuggingFaceClassifier({ modelName?, device?, pipeline?, labels? })`
  - Uses injected `pipeline` or loads `@xenova/transformers`
  - Must provide `modelName` or `pipeline` (constructor throws otherwise)
- Without `labels`, the label list comes from the model's full score list on the first call.

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

- Judge receives the full transcript (all rounds, the summary and each persona's `known_bias`) and a JSON schema with your labels as an enum.
- On success `judge_strategy` is `llm_judge` and `judge_details` / `judgeDetails` holds the judge's key agreements, key disagreements and decisive factor.
- If the judge call fails or its output is unusable, the verdict is a majority vote over the final round's valid persona responses, marked `llm_judge_fallback_error`, `llm_judge_fallback_invalid_json`, `llm_judge_fallback_invalid_label` or `llm_judge_fallback_invalid_confidence`. If the final round has no valid responses, the primary classifier result is returned under the same marker.
- If every persona call failed, the judge makes no LLM call and returns the primary classifier result as `llm_judge_fallback_personas_failed`.

#### Bayesian Judge

- Python: `BayesianJudge(persona_priors=None)`
- TypeScript: `new BayesianJudge(priors={})`

Uses persona priors/reliability maps if provided.

### Evaluating the jury

`JuryEvaluator` measures whether the jury beats your primary classifier on your own
labelled data, and what that costs. It classifies every text once, debates every item
whose primary confidence is below `band_upper` (default 0.95), and reports the primary
and jury labels side by side.

- Python: `report = await JuryEvaluator(jury).evaluate(texts, labels, band_upper=0.95, max_escalations=None, concurrency=5)`
- TypeScript: `const report = await new JuryEvaluator(jury).evaluate({ texts, labels, bandUpper: 0.95, maxEscalations, concurrency: 5 })`

`max_escalations` / `maxEscalations` guards spend: when more items fall below
`band_upper`, `TooManyEscalationsError` is raised after the primary pass and before any
debate. The jury's own threshold and `escalation_override` are ignored, and `jury.stats`
is not touched. Each debate goes through `jury.escalate(text, primary)`, a public `Jury`
method that runs the same cost gates, debate, judge and callbacks as the escalated
branch of `classify`, for a primary result you already have.

What the report gives you:

- `summary()`: `n`, `band_upper`, `primary_accuracy`, `debated`,
  `jury_accuracy_on_debated`, `primary_accuracy_on_debated`, `flips_helped` (primary
  wrong, jury right), `flips_hurt` (primary right, jury wrong), `debate_cost_usd`,
  `unpriced_calls`, `mean_debate_cost_usd`, `latency_ms_p50`, `latency_ms_p95`
  (nearest-rank, per debated item), `degraded`, `fallbacks` (counts of fallback and
  cost-guard judge strategies) and `confusion` (`primary` over all items, `jury` over
  debated items, as `{expected: {predicted: count}}`). TypeScript uses camelCase keys.
- `threshold_sweep(thresholds=None, error_cost=10.0, escalation_cost=None)` /
  `thresholdSweep({ thresholds, errorCost, escalationCost })`: one row per threshold
  with `threshold`, `escalation_rate`, `system_accuracy`, `jury_accuracy`,
  `primary_accuracy`, `errors` and `total_cost`. At threshold t, items below t take the
  jury's label and the rest keep the primary label;
  `total_cost = errors * error_cost + escalations * escalation_cost`. Leaving
  `escalation_cost` unset uses the measured mean debate cost, or 0.05 when no call was
  priced. A threshold above `band_upper` raises, because those items were never debated.
- `best_threshold(error_cost=10.0, escalation_cost=None, thresholds=None)` /
  `bestThreshold({ ... })`: the threshold with the lowest `total_cost` (the lowest one
  wins a tie).
- `items`: per text, the primary label, confidence, correctness and cost and, when
  debated, the jury's label, confidence, correctness, judge strategy, cost, duration,
  `unpriced_calls` and whether it was degraded.
- `to_dict()` / `toDict()`: `band_upper`, `summary` and `items`, with snake_case keys
  in both SDKs.

Unknown cost stays unknown. A debate whose calls reported no cost has a `None` / `null`
cost and adds to `unpriced_calls`; `debate_cost_usd` sums the known costs only and is
`None` / `null` when there are none. An item's debate cost is the verdict total minus
the primary cost, so it is also unknown when a custom classifier leaves `cost_usd`
unset (the built-in local classifiers report 0). The TypeScript `LiteLLMClient` never
reports cost, so TS debate costs are `null` unless your `llmClient` fills `costUsd`.

`examples/evaluate_jury.py` and `examples/typescript/evaluate_jury.ts` run the
evaluator offline with a stub LLM client.

### Threshold Calibration

- Python: `ThresholdCalibrator(jury)` then `await calibrate(texts, labels, error_cost=10.0, escalation_cost=None, thresholds=None, use_jury=False)`
- TypeScript: `new ThresholdCalibrator(jury)` then `await calibrate({ texts, labels, errorCost=10, escalationCost?, thresholds?, useJury=false })`

Both classify each text once and pick the threshold with the lowest
`errors * error_cost + escalations * escalation_cost`. An item escalates at threshold t
when its confidence is below t or is not a finite number, the same rule `Jury` uses.

- Default mode never runs the jury. Each escalation costs `escalation_cost` (default
  0.05) and counts as neither right nor wrong, so `accuracy` is the primary
  classifier's accuracy on the items it keeps. This mode is cheap but assumes nothing
  about whether the jury helps.
- `use_jury=True` / `useJury: true` runs `JuryEvaluator` with `band_upper` set to the
  highest candidate threshold (one debate per item below it) and picks the threshold
  from the measured sweep, so a jury that gets items wrong pulls the threshold down.
  `escalation_cost` then defaults to the measured mean debate cost. The evaluation is
  kept on `calibrator.evaluation_report` / `calibrator.evaluationReport`.

Report:

- Python: `calibration_report()`
- TypeScript: `calibrationReport()`

Both return the best threshold, whether the jury was used, and rows with threshold,
accuracy, escalation rate and total cost. After a jury calibration the rows also carry
`system_accuracy` and `jury_accuracy`, and the report adds the evaluation `summary`.
`calibrate(...)` mutates `jury.threshold` to the best threshold.

Default threshold candidates when not provided:

- `0.50, 0.55, ..., 0.95`

### LLM Transport (`LiteLLMClient`)

#### Python

- `LiteLLMClient(timeout_seconds=60.0, max_attempts=3, api_key=None, api_base=None)`; pass it to `Jury(llm_client=...)`
- `timeout_seconds` bounds each request (`None` keeps litellm's own default); `max_attempts` counts the first try; `api_key` and `api_base` are passed to litellm when set
- `complete(model, system_prompt, prompt, temperature=0.0, response_format=None)` calls `litellm.acompletion`
- Returns: `{content, tokens, cost_usd}` (`cost_usd` is `None` when litellm cannot price the model)
- Raises a runtime error if `litellm` is not installed and no custom `llm_client` is injected.

#### TypeScript

- `new LiteLLMClient({ baseUrl?, apiKey?, timeoutMs?, maxAttempts?, logger? })`; `timeoutMs` defaults to 60000 per attempt and `maxAttempts` to 3 (first try included)
- Falls back to env vars:
  - `LITELLM_BASE_URL`, `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
  - `LITELLM_API_KEY`, `OPENAI_API_KEY`
- Sends `POST /chat/completions`
- Returns `{content, tokens, costUsd}`. `costUsd` is always `undefined` because no viable npm cost-estimation library exists; a custom `llmClient` can provide cost data if needed.
- Throws `No API key configured. Set LITELLM_API_KEY or OPENAI_API_KEY, or inject a custom llmClient.` before sending a request when there is no key.
- The client `Jury` creates by default logs retries through the Jury's `logger`.

Both clients retry connection errors, timeouts, 429 and 5xx responses, honour `Retry-After` (capped at 60 s), and omit temperature for reasoning models (`gpt-5*`, `o1*`, `o3*`, with or without a provider prefix).

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

## Troubleshooting

The most common gotchas across both SDKs. For the full list with code examples, see the per-package READMEs:
[Python](packages/python/README.md#troubleshooting) · [TypeScript](packages/typescript/README.md#troubleshooting).

| Symptom | Likely cause | Fix |
|---|---|---|
| Auth / 401 on first LLM call | `OPENAI_API_KEY` not set, or wrong provider for the model | `export OPENAI_API_KEY=...` or pass an explicit client (`llm_client=LiteLLMClient(api_key=...)` / `llmClient: new LiteLLMClient({ apiKey })`) |
| Primary result has `confidence` 0 and the item always escalates | `LLMClassifier` could not use the model's reply; `primary_result.raw_output["error"]` / `primaryResult.rawOutput.error` says why (`invalid_json`, `label_not_in_labels`, `invalid_confidence`) | Use a model that honours `response_format` JSON schemas; or wrap your own call in `FunctionClassifier` to control parsing |
| Repeated 429s after retries | Rate-limit budget exhausted after 3 attempts per call (the SDK already waits for `Retry-After`, up to 60 s) | Lower `debate_concurrency` / `debateConcurrency` and batch `concurrency`; raise `max_attempts` / `maxAttempts`; use a higher-tier key |
| `judge_strategy` / `judgeStrategy` is `cost_guard_pre_flight` | Pre-flight estimate exceeded `max_debate_cost_usd`; no debate ran. The estimate counts the summariser and LLM judge calls too, so a cap set for an older version can now trip | Raise the cap, lower `max_rounds`, or lower `estimated_cost_per_persona_usd` if your calls cost less |
| `judge_strategy` / `judgeStrategy` is `cost_guard_primary_fallback` | Spend during the debate hit the cap. Calls that reported no cost are charged at `estimated_cost_per_persona_usd` | Same as above; spend can still overshoot by up to one concurrency batch |
| `judge_strategy` / `judgeStrategy` is `cost_guard_user_override` | Your `on_cost_estimate` / `onCostEstimate` callback returned False | Working as intended: the debate was skipped per your policy |
| `judge_strategy` / `judgeStrategy` is `llm_judge_fallback_error` | The LLM judge call failed (after retries) | The verdict is a majority vote over the final round. Check the judge model and key; consider routing these to review |
| `judge_strategy` / `judgeStrategy` is `llm_judge_fallback_invalid_json`, `_invalid_label` or `_invalid_confidence` | The judge replied with unparseable JSON, a label outside your labels, or a non-numeric confidence | Same majority-vote fallback. Use a judge model that honours `response_format` |
| `judge_strategy` / `judgeStrategy` is `llm_judge_fallback_personas_failed` | Every persona call failed, so there was nothing to judge | The primary classifier result is returned; see `debate_degraded` below |
| Verdict is never escalated even at low confidence | `personas=[]` silently disables escalation (by design) | Pass at least one persona |
| `debate_degraded` / `debateDegraded` is true | One or more persona calls failed (auth, rate-limit exhaustion, unparseable output, a label outside your labels). Failed responses stay in the transcript with `failed=True` and carry no vote; if the whole final round failed, the primary classifier result is returned | Inspect `persona_failures` / `personaFailures` and the transcript's `failed` responses; consider routing degraded verdicts to human review |
| `total_cost_usd` / `totalCostUsd` is `None` / `null` | Python: the model is not in litellm's pricing table, or the primary classifier reported no cost. TypeScript: the default client never reports cost | Python: pin to a known-priced model. TS: inject a custom `llmClient` that fills `costUsd` |
| `total_cost_usd` looks too low | Some calls reported no cost; `debate_transcript.unpriced_calls` counts them and the total is a lower bound | Same as above |
| TypeScript logs are silent | TS `Jury` defaults to `NOOP_LOGGER` | `new Jury({ ..., logger: console })` |
| A call hangs about 60 s, then aborts | Default request timeout is 60 s per attempt in both SDKs (timeouts are retried) | `new LiteLLMClient({ timeoutMs: 30_000 })` / `LiteLLMClient(timeout_seconds=30)` |

Known problems and their status are tracked in [`docs/REVIEW.md`](docs/REVIEW.md).

## CLI (Secondary)

The product is SDK-first. CLI is provided for batch workflows.

Both packages install an `llm-jury` command: `pip install llm-jury-classifier`, or `npm install @llm-jury/core` and run `npx llm-jury`. Both read JSONL and write the same snake_case verdict rows.

### Commands

- `llm-jury classify`
- `llm-jury calibrate`
- `llm-jury eval`

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

### Input rows and exit codes

- With `--classifier function` (the default) the CLI replays predictions stored in the input, so every row needs `text`, `predicted_label` and `predicted_confidence` (a number from 0 to 1). The ground-truth `label` field is never read as a prediction, and rows that repeat a text must repeat its prediction.
- Bad usage exits with code 2 before any LLM call: a missing or out-of-range option value, an unknown `--debate-mode`, or input rows without their predictions.
- `classify` writes a `{"text", "error"}` row for each input that failed and keeps the verdicts that succeeded. It exits with code 1 only when every row failed.

### Calibrate-only options

- `--input` (required, must include ground-truth `label` per row)
- `--error-cost` default `10.0`
- `--escalation-cost` default `0.05`, or the measured mean debate cost with `--use-jury`
- `--initial-threshold` default `0.7`
- `--use-jury` runs the jury on every row below the highest threshold and calibrates on
  its measured outcomes (this makes LLM calls). Without it the jury options above have
  no effect, and `calibrate` says so on stderr when you pass any of them.

### Eval-only options

`llm-jury eval` measures the jury against the primary classifier and prints one JSON
line with `best_threshold`, `summary` and `sweep` (see
[Evaluating the jury](#evaluating-the-jury)). It takes the same input as `calibrate`.

- `--input` (required, must include ground-truth `label` per row)
- `--output` also writes the full report, with per-row results, to a JSON file
- `--band-upper` default `0.95`: debate every row whose primary confidence is below it
- `--max-escalations` stops before any debate when more rows would be debated
- `--thresholds` comma-separated, each at most `--band-upper` (default
  `0.5,0.55,...,0.95` up to `--band-upper`)
- `--error-cost` default `10.0`
- `--escalation-cost` default: the measured mean debate cost
- `--concurrency` default `5`: rows classified or debated at once

## License

MIT
