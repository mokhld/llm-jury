# @llm-jury/core

**When your classifier is uncertain, let a configurable jury of LLM personas debate and return an auditable verdict.**

[![npm](https://img.shields.io/npm/v/@llm-jury/core)](https://www.npmjs.com/package/@llm-jury/core)
[![Node.js 22.6+](https://img.shields.io/badge/node.js-22.6%2B-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

## Overview

`@llm-jury/core` is an SDK, not a hosted API. Your app imports it directly:

```ts
import { Jury, PersonaRegistry } from "@llm-jury/core";
```

It wraps a classifier returning `[label, confidence]` and adds confidence-based escalation:

1. Run primary classifier (fast path)
2. Return directly when confidence is high
3. Escalate low-confidence cases to persona debate
4. Consolidate with a judge strategy
5. Return verdict + audit trail

### Research Inspiration

`llm-jury` is inspired by the CEJ (Collaborative Expert Judgment) module described in [arXiv:2512.23732](https://arxiv.org/abs/2512.23732). This package generalizes that pattern into a domain-agnostic SDK with pluggable classifiers, multiple debate modes, multiple judge strategies, threshold calibration, and Python + TypeScript distributions.

## Install

```bash
npm install @llm-jury/core
```

## Prerequisites

- Node.js `>=22.6`
- For real LLM calls: `OPENAI_API_KEY` (or provider key through your LiteLLM/OpenAI setup)

## Quick Start

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

The default LLM client sends requests to `POST /chat/completions` on `OPENAI_BASE_URL` / `LITELLM_BASE_URL` / `https://api.openai.com/v1`.

## SDK Response

`jury.classify(text)` returns a `Verdict`. There are two shapes depending on whether the input was escalated.

### Fast path (confidence above threshold)

When the primary classifier is confident enough, the verdict is returned directly with no debate.

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

| Field | Type | Description |
|---|---|---|
| `label` | `string` | Final classification |
| `confidence` | `number` | Final confidence (0.0-1.0) |
| `reasoning` | `string` | Human-readable explanation |
| `wasEscalated` | `boolean` | Whether debate was triggered |
| `primaryResult` | `ClassificationResult` | Fast-path classifier output |
| `debateTranscript` | `DebateTranscript \| null` | Full debate audit trail incl. `rounds`, `summary`, token/cost totals (null if not escalated) |
| `judgeStrategy` | `string` | Strategy that produced the verdict |
| `totalDurationMs` | `number` | Wall-clock time (ms) |
| `totalCostUsd` | `number \| null` | API cost in USD |

### Persona response fields

| Field | Type | Description |
|---|---|---|
| `personaName` | `string` | Which persona |
| `label` | `string` | This persona's classification |
| `confidence` | `number` | This persona's confidence |
| `reasoning` | `string` | Full reasoning chain |
| `keyFactors` | `string[]` | Key decision factors |
| `dissentNotes` | `string \| null` | Rebuttal in deliberation/adversarial modes |
| `tokensUsed` | `number` | Tokens consumed |
| `costUsd` | `number \| null` | API cost for this call |

`DebateTranscript` also includes `summary` (`string?`) — a structured summary produced during the Summarisation stage of the deliberation pipeline (undefined in non-deliberation modes).

## Choosing What To Use

### Classifiers

| Classifier | When to use | Example |
|---|---|---|
| `FunctionClassifier` | Wrap an existing model or function | `new FunctionClassifier(fn, labels)` |
| `LLMClassifier` | Primary classifier is an LLM | `new LLMClassifier({ labels: ["safe","unsafe"] })` |
| `HuggingFaceClassifier` | Local HuggingFace model | `new HuggingFaceClassifier({ modelName: "..." })` |
| `SklearnClassifier` | Wrap an sklearn-like model | `new SklearnClassifier(model, labels, vectorizer)` |

### Built-in Persona Sets

| Method | Domain | Personas |
|---|---|---|
| `PersonaRegistry.contentModeration()` | Trust & Safety | Policy Analyst, Cultural Context Expert, Harm Assessment Specialist |
| `PersonaRegistry.legalCompliance()` | Legal/Regulatory | Regulatory Attorney, Business Risk Analyst, Industry Standards Expert |
| `PersonaRegistry.medicalTriage()` | Healthcare | Clinical Safety Reviewer, Contextual Historian, Resource Allocation Analyst |
| `PersonaRegistry.financialCompliance()` | AML/KYC | AML Investigator, Risk Quant, Business Controls Reviewer |
| `PersonaRegistry.custom([...])` | Any domain | Provide your own persona objects |

### Judge Strategies

| Strategy | How it decides | Best for |
|---|---|---|
| `new MajorityVoteJudge()` | Counts persona votes. Confidence = fraction agreeing. | Fast, no extra LLM call |
| `new WeightedVoteJudge()` | Weights votes by persona confidence. | When confidence scores vary significantly |
| `new LLMJudge()` | LLM reads full transcript and synthesises verdict. | Maximum quality, auditable reasoning |
| `new BayesianJudge()` | Bayesian aggregation with optional persona priors. | When you have reliability data on personas |

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
- **Default debate mode is deliberation** for maximum value — it runs the full 4-stage CEJ pipeline. For cheaper/faster operation, use `{ mode: DebateMode.Independent }`.
- **Cost tracking** — `totalCostUsd` is always `undefined` unless a custom `llmClient` provides cost data (no viable npm cost-estimation library exists).
- **Empty personas disables escalation**: If you pass `personas: []`, the jury always returns the primary classifier result.

## API Reference

### Public Exports

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

| Option | Default | Description |
|---|---|---|
| `classifier` | (required) | Primary classifier |
| `personas` | (required) | List of personas |
| `confidenceThreshold` | `0.7` | Escalation threshold |
| `judge` | defaults to `LLMJudge` | Judge strategy |
| `debateConfig` | `undefined` | Debate configuration |
| `escalationOverride` | `undefined` | Force escalation |
| `maxDebateCostUsd` | `undefined` | Cost cap for debate |
| `estimatedCostPerPersonaUsd` | `0.01` | Heuristic per-call cost used for the pre-flight estimate |
| `debateConcurrency` | `5` | Max concurrent persona calls |
| `onEscalation` | `undefined` | Fires when input is escalated to debate. `(text, primaryResult) => void` |
| `onCostEstimate` | `undefined` | Fires with `(estimatedMaxDebateCostUsd, text)` immediately before a debate would run. Return `false` to skip the debate (verdict marked `cost_guard_user_override`); return `true` / `undefined` to proceed. |
| `onVerdict` | `undefined` | Verdict callback |
| `llmClient` | `undefined` | LLM transport override |

Methods:

- `await classify(text)` — classify a single input
- `await classifyBatch(texts, concurrency=10)` — classify multiple inputs

Behavior notes:

- Escalation condition is strictly `< threshold` (exactly equal does not escalate).
- If `personas` is empty, jury escalation is effectively disabled.
- If `maxDebateCostUsd` is exceeded, result falls back to primary classifier with `judgeStrategy` set to `cost_guard_primary_fallback`.
- `Jury.estimatedMaxDebateCostUsd` (getter) returns the heuristic upper-bound estimate `personas.length × maxRounds × estimatedCostPerPersonaUsd`. Useful for budgeting before any call.
- `onCostEstimate` runs after the escalation decision but before any LLM call for the debate, *and* before the `maxDebateCostUsd` guard. Lets you layer per-tenant budgets, time-of-day gates, etc. on top of the hard cap.

Stats: `jury.stats.total`, `fastPath`, `escalated`, `escalationRate`, `costSavingsVsAlwaysEscalate`.

### `DebateConfig` Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `deliberation` | Debate mode |
| `maxRounds` | `2` | Max deliberation rounds |
| `includePrimaryResult` | `true` | Include primary result in prompts |
| `includeConfidence` | `true` | Include confidence in prompt context |
| `earlyStopMinConfidence` | `undefined` | **F7:** opt-in high-confidence early stop for DELIBERATION mode. When set, the deliberation loop exits early after any round whose **minimum** persona confidence is `>=` this value, even if personas disagree on label. Unanimous-label consensus still triggers early exit regardless. undefined = original behaviour (unanimous-label only). |

### Personas

Persona fields: `name`, `role`, `systemPrompt`, `model="gpt-5-mini"`, `temperature=0.3`, `knownBias?`.

### Classifiers (API)

All classifiers implement `classify(text)` and expose `labels`.

- **FunctionClassifier**: `new FunctionClassifier(fn, labels)` where `fn` may return tuple or Promise tuple
- **LLMClassifier**: `new LLMClassifier({ model, labels, systemPrompt, llmClient, temperature })` — expects model JSON response with `label` and `confidence`; falls back to first label with `confidence=0` on parse failure
- **SklearnClassifier**: `new SklearnClassifier(model, labels, vectorizer?)` where model has `predictProba(...)`
- **HuggingFaceClassifier**: `new HuggingFaceClassifier({ modelName?, device?, pipeline? })` — uses injected `pipeline` or loads `@xenova/transformers`; must provide `modelName` or `pipeline`

### Judge Strategies (API)

- **MajorityVoteJudge**: `new MajorityVoteJudge()` — confidence = fraction of personas voting winning label
- **WeightedVoteJudge**: `new WeightedVoteJudge()` — confidence based on confidence-weighted label scores
- **LLMJudge**: `new LLMJudge({ model, systemPrompt, temperature, llmClient })` — falls back to primary result with `llm_judge_fallback_invalid_json` if JSON parse fails
- **BayesianJudge**: `new BayesianJudge(priors={})` — uses persona priors/reliability maps if provided

### Threshold Calibration

`new ThresholdCalibrator(jury)` then `await calibrate({ texts, labels, errorCost=10, escalationCost=0.05, thresholds? })`.

Report: `calibrationReport()` returns rows with threshold, accuracy, escalation rate, and total cost. `calibrate(...)` mutates `jury.threshold` to the best threshold.

### LLM Transport (`LiteLLMClient`)

- `new LiteLLMClient({ baseUrl?, apiKey?, timeoutMs? })`
- Falls back to env vars: `LITELLM_BASE_URL`, `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`); `LITELLM_API_KEY`, `OPENAI_API_KEY`
- Sends `POST /chat/completions`
- Returns `{ content, tokens, costUsd }` — `costUsd` is always `undefined` (no viable npm cost-estimation library)
- Throws before request if no API key is configured

Temperature is automatically omitted for reasoning models (`gpt-5*`, `o1*`, `o3*`).

### Response Cache (`CachingLLMClient`)

Opt-in LRU wrapper around any `LLMClient`. Keyed on
`(model, systemPrompt, prompt, temperature, responseFormat)`.
Successful responses only — rejections propagate without being cached.

```ts
import { CachingLLMClient, Jury, LiteLLMClient } from "@llm-jury/core";

const jury = new Jury({
  // ...
  llmClient: new CachingLLMClient(new LiteLLMClient(), {
    maxSize: 1000,      // LRU cap
    ttlSeconds: 3600,   // optional; omit for no expiry
  }),
});
```

`hits`, `misses`, and `size` are exposed for introspection. Call
`clear()` to drop everything. The cache is in-process and per-instance;
share the `CachingLLMClient` object across `Jury` instances if you want
a shared cache. Caches at any temperature — if you need fresh
stochastic samples, don't wrap.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error: LiteLLMClient requires an API key` (thrown before HTTP) | `OPENAI_API_KEY` and `LITELLM_API_KEY` both unset, and no `apiKey` passed | `export OPENAI_API_KEY=...` or `new LiteLLMClient({ apiKey: "..." })` |
| `Authentication`/`401` on first LLM call | API key present but wrong / wrong provider for the model | Verify the key against the provider; check `OPENAI_BASE_URL` / `LITELLM_BASE_URL` |
| Hangs ~60s then aborts | Default `timeoutMs=60_000`; provider didn't respond in time | `new LiteLLMClient({ timeoutMs: 30_000 })` — covered in T7 timeout test |
| Verdict's `label` is the first label with `confidence=0` | `LLMClassifier` couldn't parse the model's JSON response | Use a model that supports `response_format`; persona responses are schema-constrained (F2), but `LLMClassifier`'s own parse path is not yet — see audit S3 |
| `Error: HTTP 429 ...` after retries | Rate-limit budget exhausted; client retries 3× with backoff and now detects 429 / 5xx via `err.status` (B9 fix), but doesn't honour `Retry-After` (R7) | Lower `debateConcurrency`; lower batch `concurrency`; use a higher-tier key |
| `verdict.judgeStrategy === "cost_guard_pre_flight"` (no debate ran) | Pre-flight estimate exceeded `maxDebateCostUsd` | Raise the cap, lower `maxRounds`, or accept the primary classifier verdict |
| `verdict.judgeStrategy === "cost_guard_primary_fallback"` (debate ran partially) | Actual mid-debate spend hit the cap | Same as above; can still overshoot by up to one concurrency-batch (in-flight calls aren't cancellable) |
| `verdict.judgeStrategy === "cost_guard_user_override"` | Your `onCostEstimate` callback returned `false` | Working as intended — the debate was skipped per your policy |
| `verdict.totalCostUsd` is `undefined` / `null` even after a debate | Default `LiteLLMClient` cannot estimate cost (no npm pricing library) | Inject a custom `llmClient` that fills `costUsd`; the SDK forwards it through |
| `classifyBatch` returns fewer / duplicated results than inputs | This was B1 (fixed) — if you see it on a current version, file a bug | Pin to the latest version; share repro under the audit's `classifyBatch` test pattern |
| Verdict is never escalated even at very low confidence | `personas: []` silently disables escalation (by design) | Pass at least one persona |
| One persona always missing from `verdict.debateTranscript.rounds` | That persona's `model` is invalid / not available to your key. Single-persona failure no longer crashes the verdict (B2 fix) — it's dropped | Pass a `logger` (e.g. `console`) to `new Jury({ ..., logger: console })` to see why |
| Debate summary is `undefined` even in deliberation mode | Summariser LLM call failed; persona rounds are still load-bearing (post-D6 fix) | Same — pass a `logger` to see the warning |
| Logs are silent in production | Default logger is `NOOP_LOGGER` (parity with Python being opt-in) | `new Jury({ ..., logger: console })` or pass any object matching the `Logger` interface |

For deeper context on known open issues, see `AUDIT.md` §3 (Reliability) and §1 (Bugs).

## Examples

Runnable examples live in `examples/typescript/` at the repo root (require `OPENAI_API_KEY` except `threshold_calibration`):

```bash
node --experimental-strip-types examples/typescript/content_moderation.ts
node --experimental-strip-types examples/typescript/custom_personas.ts
node --experimental-strip-types examples/typescript/legal_compliance.ts
node --experimental-strip-types examples/typescript/threshold_calibration.ts
```

Each example imports from `@llm-jury/core` to mirror real user code — `npm install @llm-jury/core` in your project first, or `npm link` the local package when running directly from a clone.

## Testing

```bash
npm test
```

### Real API Smoke Test

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" node --test --experimental-strip-types tests/smoke/real-api.test.ts
```

## CLI

The CLI is for batch workflows. The primary interface is the TypeScript API above.

```bash
npm run build
node dist/cli/main.js classify \
  --input input.jsonl \
  --output verdicts.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --judge-model gpt-5-mini \
  --persona-model gpt-5-mini \
  --threshold 0.7 \
  --labels safe,unsafe
```

Calibration:

```bash
node dist/cli/main.js calibrate \
  --input calibration.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --judge-model gpt-5-mini \
  --persona-model gpt-5-mini \
  --labels safe,unsafe
```

Supported classifier specs: `function`, `llm:<model>`, `huggingface:<model>`.

## License

MIT
