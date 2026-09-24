# @llm-jury/core

**When your classifier is uncertain, let a configurable jury of LLM personas debate and return an auditable verdict.**

[![npm](https://img.shields.io/npm/v/@llm-jury/core)](https://www.npmjs.com/package/@llm-jury/core)
[![Node.js 20+](https://img.shields.io/badge/node.js-20%2B-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/mokhld/llm-jury/blob/main/LICENSE)

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

- Node.js `>=20` (ES modules only; `require()` works on Node 20.19+ and 22.12+). Running the TypeScript examples directly needs Node `>=22.6` for `--experimental-strip-types`.
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

`jury.classify(text)` returns a `Verdict`. There are two shapes depending on whether the input was escalated. The JSON below is what `verdict.toDict()` (and `JSON.stringify(verdict)`) returns. The samples are illustrative and shortened (persona `rawResponse` is left out). The durations and costs are made up; real escalations take far longer and cost more (see [Important Notes](#important-notes) for measured numbers). With the default `LiteLLMClient`, LLM calls report no cost: persona `costUsd` is absent, `unpricedCalls` counts every call and `totalCostUsd` is `null`.

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

When confidence is too low, the input goes through persona debate and a judge produces the final verdict. In this sample the three personas agree in the opening round, so the debate stops there with no second round and no summary. The costs shown assume a custom `llmClient` that reports them.

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

### Verdict field reference

| Field | Type | Description |
|---|---|---|
| `label` | `string` | Final classification |
| `confidence` | `number` | Final confidence (0.0 to 1.0) |
| `reasoning` | `string` | Human-readable explanation |
| `wasEscalated` | `boolean` | Whether debate was triggered |
| `primaryResult` | `ClassificationResult` | Fast-path classifier output |
| `debateTranscript` | `DebateTranscript \| null` | Full debate audit trail (see below); null if no debate ran |
| `judgeStrategy` | `string` | Strategy that produced the verdict, including the fallback markers listed under [Troubleshooting](#troubleshooting) |
| `totalDurationMs` | `number` | Wall-clock time (ms) |
| `totalCostUsd` | `number \| null` | Primary classifier plus debate and judge cost in USD. `null` when any part is unknown (always, with the default client); a lower bound when `debateTranscript.unpricedCalls > 0` |
| `personaFailures` | `number` | Persona calls across the debate that failed (LLM error, unparseable output, or a label outside the configured set) |
| `debateDegraded` | `boolean` | Getter (also in `toDict()`): true when `personaFailures > 0`, so the verdict was decided by fewer jurors than configured. Useful for routing to human review |
| `judgeDetails` | `Record<string, unknown> \| null` | `LLMJudge` only: `keyAgreements`, `keyDisagreements`, `decisiveFactor`. Null for other judges and fallbacks |
| `libraryVersion` | `string` | `@llm-jury/core` version that produced the verdict |
| `createdAt` | `string` | ISO 8601 UTC timestamp |

### Debate transcript fields

| Field | Type | Description |
|---|---|---|
| `inputText` | `string` | The text that was classified |
| `primaryResult` | `ClassificationResult` | Primary classifier output |
| `rounds` | `PersonaResponse[][]` | One array per round, in order |
| `summary` | `string?` | Summariser output in deliberation mode. Undefined when the debate stopped early, in other modes, or when the summariser call failed |
| `durationMs` | `number` | Debate wall-clock time (ms) |
| `totalTokens` | `number` | Tokens used by persona and summariser calls |
| `totalCostUsd` | `number \| null` | Sum of the persona and summariser calls that reported a cost; null when none did |
| `unpricedCalls` | `number?` | Debate calls that reported no cost. Non-zero means `totalCostUsd` is a lower bound |
| `personaBiases` | `Record<string, string>?` | Persona name to `knownBias`, for personas that declare one. The LLM judge sees these |

`countPersonaFailures(transcript.rounds)` returns the number of failed responses across all rounds.

### Persona response fields

| Field | Type | Description |
|---|---|---|
| `personaName` | `string` | Which persona |
| `label` | `string` | This persona's classification |
| `confidence` | `number` | This persona's confidence |
| `reasoning` | `string` | Full reasoning chain |
| `keyFactors` | `string[]` | Key decision factors |
| `dissentNotes` | `string?` | Rebuttal in deliberation/adversarial modes |
| `rawResponse` | `string?` | The model's raw reply |
| `tokensUsed` | `number?` | Tokens consumed |
| `costUsd` | `number?` | API cost for this call; absent when the client reported none |
| `failed` | `boolean?` | True when this response is a placeholder for a failed persona call. Failed responses stay in the transcript for audit but carry no vote |

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

- **Temperature is handled automatically.** The SDK omits the temperature parameter for reasoning models (`gpt-5*`, `o1*`, `o3*`, also behind a provider prefix such as `openai/gpt-5-mini`). No configuration needed.
- **Escalation is strictly `< threshold`**: confidence exactly equal to the threshold does NOT escalate. A missing or non-numeric primary confidence (`NaN`, a string, `undefined`) always escalates.
- **Automatic retry**: each LLM call gets 3 attempts in total (the first try plus 2 retries) on network errors, timeouts, 429 and 5xx responses; other errors fail at once. The client waits for the provider's `Retry-After` header when it sends one (capped at 60 s) and backs off exponentially otherwise. Change the count with `new LiteLLMClient({ maxAttempts })`.
- **Default debate mode is deliberation**, the full 4-stage CEJ pipeline. For cheaper and faster runs use `new DebateConfig({ mode: DebateMode.INDEPENDENT })`.
- **Deliberation stops early** after any round, the opening one included, when the personas' labels are unanimous or `earlyStopMinConfidence` is met. A debate that stops early has no summary.
- **Latency and cost of an escalation**: a debate makes several rounds of LLM calls. Live runs in February 2026 with the default `gpt-5-mini`, three personas and a mix of debate modes and judges took 27 to 57 s and cost $0.007 to $0.015 per escalated item. Use that as a rough guide only; your models, personas and inputs will change it. The fast path costs one primary classifier call.
- **Cost tracking**: the default `LiteLLMClient` never reports cost, so `totalCostUsd` is `null` unless a custom `llmClient` returns `costUsd`. Unknown cost is `null`, never 0. An escalated `totalCostUsd` includes the primary classifier, is `null` when the primary cost or the whole debate cost is unknown, and is a lower bound when `debateTranscript.unpricedCalls > 0`.
- **The cost cap is checked twice.** Before a debate, `estimatedMaxDebateCostUsd` is compared with `maxDebateCostUsd`; the estimate counts every persona call in every round, the summariser in deliberation mode and the judge when it is an `LLMJudge`, each at `estimatedCostPerPersonaUsd` (default $0.01 per call). During the debate, reported spend is compared with the cap, and calls that reported no cost (every call, with the default client) are charged at that same per-call estimate.
- **Empty personas disables escalation**: If you pass `personas: []`, the jury always returns the primary classifier result.
- **Untrusted input**: every prompt fences the input in `<input>` tags marked as untrusted data, and labels returned by models are checked against your labels. See [Prompt injection and untrusted input](https://github.com/mokhld/llm-jury#prompt-injection-and-untrusted-input) for what callers should still do.

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
  CachingLLMClient,
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
| `estimatedCostPerPersonaUsd` | `0.01` | Estimated cost of one LLM call (persona, summariser or judge). Used for the pre-flight estimate and charged against the cap for calls that report no cost |
| `debateConcurrency` | `5` | Max concurrent persona calls |
| `onEscalation` | `undefined` | Fires when input is escalated to debate. `(text, primaryResult) => void` |
| `onCostEstimate` | `undefined` | Fires with `(estimatedMaxDebateCostUsd, text)` immediately before a debate would run. Return `false` to skip the debate (verdict marked `cost_guard_user_override`); return `true` / `undefined` to proceed. |
| `onVerdict` | `undefined` | Fires once with every verdict `classify` returns, including fast-path and cost-guard verdicts. `(verdict) => void` |
| `llmClient` | `undefined` | LLM transport override |
| `logger` | `NOOP_LOGGER` (silent) | Any object with `debug`, `info`, `warn` and `error` methods, such as `console`. Also passed to the default `LiteLLMClient` and `LLMJudge` |

Methods:

- `await classify(text)`: classify a single input
- `await classifyBatch(texts, concurrency = 10, returnExceptions = false)`: classify multiple inputs. With `returnExceptions: true`, a failing text yields its `Error` in-slot instead of rejecting the whole batch. Without it, the first failure rejects the batch and no further text starts a debate.

Behavior notes:

- Escalation condition is strictly `< threshold` (exactly equal does not escalate).
- If `personas` is empty, jury escalation is effectively disabled.
- Failed persona calls (LLM error, unparseable output, or a label outside the configured set) are kept in the transcript as placeholders with `failed: true` but carry no vote. If the whole final round failed, judges return the primary classifier result. `Verdict.personaFailures` counts them and `Verdict.debateDegraded` is true when any persona failed; use it to route degraded verdicts to human review.
- `Jury.estimatedMaxDebateCostUsd` (getter) is the pre-flight estimate: (persona calls + 1 summariser call in deliberation mode + 1 judge call for an `LLMJudge`) x `estimatedCostPerPersonaUsd`. Persona calls are `personas.length x maxRounds` in deliberation mode and `personas.length` in the other modes. If it exceeds `maxDebateCostUsd`, no debate runs and `judgeStrategy` is `cost_guard_pre_flight`.
- If spend during the debate exceeds `maxDebateCostUsd`, the result falls back to the primary classifier with `judgeStrategy` set to `cost_guard_primary_fallback`. Calls that reported no cost are charged at `estimatedCostPerPersonaUsd`.
- `onCostEstimate` runs after the escalation decision but before any LLM call for the debate, *and* before the `maxDebateCostUsd` guard. Lets you layer per-tenant budgets, time-of-day gates, etc. on top of the hard cap.

Stats: `jury.stats.total`, `fastPath`, `escalated`, `escalationRate`, `costSavingsVsAlwaysEscalate`.

### `DebateConfig` Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `deliberation` | Debate mode |
| `maxRounds` | `2` | Max deliberation rounds |
| `includePrimaryResult` | `true` | Include primary result in prompts |
| `includeConfidence` | `true` | Include confidence in prompt context |
| `earlyStopMinConfidence` | `undefined` | Opt-in early stop for deliberation mode. When set, the debate ends after any round, the opening one included, whose **lowest** persona confidence is `>=` this value, even if personas disagree on label. Unanimous labels end it regardless. Undefined means only unanimous labels stop early. |

### Personas

Persona fields: `name`, `role`, `systemPrompt`, `model="gpt-5-mini"`, `temperature=0.3`, `knownBias?`.

### Classifiers (API)

All classifiers implement `classify(text)` and expose `labels`.

- **FunctionClassifier**: `new FunctionClassifier(fn, labels)` where `fn` may return tuple or Promise tuple
- **LLMClassifier**: `new LLMClassifier({ model, labels, systemPrompt, llmClient, temperature })`. `labels` must hold at least one label. Sends a JSON schema with the labels as an enum; the returned label is matched to yours (exact, then case-insensitive). Output it cannot use returns confidence 0 (so the jury escalates it) with the reason in `rawOutput.error`: `invalid_json`, `label_not_in_labels` or `invalid_confidence`.
- **SklearnClassifier**: `new SklearnClassifier(model, labels, vectorizer?)` where model has `predictProba(...)`. Columns are named by `model.classes` when those are the same set as `labels`, otherwise by position; a label count that differs from `classes` throws.
- **HuggingFaceClassifier**: `new HuggingFaceClassifier({ modelName?, device?, pipeline?, labels? })`. Uses the injected `pipeline` or loads `@xenova/transformers`; must provide `modelName` or `pipeline`. Without `labels`, the label list comes from the model's full score list on the first call.

### Judge Strategies (API)

- **MajorityVoteJudge**: `new MajorityVoteJudge()`. Confidence is the fraction of the final round's valid responses voting for the winning label.
- **WeightedVoteJudge**: `new WeightedVoteJudge()`. Confidence comes from confidence-weighted label scores.
- **LLMJudge**: `new LLMJudge({ model, systemPrompt, temperature, llmClient, logger })`. Reads every round, the summary and each persona's `knownBias`, and answers with a label-enum JSON schema. On success `judgeStrategy` is `llm_judge` and `verdict.judgeDetails` holds `keyAgreements`, `keyDisagreements` and `decisiveFactor`. When its call fails or its output is unusable, it returns a majority vote over the final round (`llm_judge_fallback_error`, `llm_judge_fallback_invalid_json`, `llm_judge_fallback_invalid_label`, `llm_judge_fallback_invalid_confidence`), or the primary result if that round has no valid responses. When every persona failed it skips its call and returns the primary result (`llm_judge_fallback_personas_failed`).
- **BayesianJudge**: `new BayesianJudge(priors={})`. Uses persona priors/reliability maps if provided.

### Evaluating the jury

`JuryEvaluator` measures whether the jury beats your primary classifier on your own labelled data, and what that costs:

```ts
import { JuryEvaluator } from "@llm-jury/core";

const report = await new JuryEvaluator(jury).evaluate({
  texts,
  labels,
  bandUpper: 0.95, // debate every item whose primary confidence is below this
  maxEscalations: 200, // throw TooManyEscalationsError before any debate if more would run
  concurrency: 5,
});
report.summary(); // accuracy, flipsHelped / flipsHurt, debate cost, latency, confusion
report.thresholdSweep(); // per threshold: escalationRate, systemAccuracy, juryAccuracy, totalCost
report.bestThreshold({ errorCost: 10 });
```

The primary classifier runs once per text. Items below `bandUpper` go to `jury.escalate(text, primary)`, which runs the same cost gates, debate, judge and callbacks as the escalated branch of `classify`; the jury's own threshold is ignored and `jury.stats` is not touched.

- `summary()` fields: `n`, `bandUpper`, `primaryAccuracy`, `debated`, `juryAccuracyOnDebated`, `primaryAccuracyOnDebated`, `flipsHelped` (primary wrong, jury right), `flipsHurt` (primary right, jury wrong), `debateCostUsd`, `unpricedCalls`, `meanDebateCostUsd`, `latencyMsP50`, `latencyMsP95`, `degraded`, `fallbacks`, `confusion` (`primary` over all items, `jury` over debated items).
- `thresholdSweep({ thresholds?, errorCost = 10, escalationCost? })`: at threshold t, items below t take the jury's label and the rest keep the primary label; `totalCost = errors * errorCost + escalations * escalationCost`. Leaving `escalationCost` unset uses the measured mean debate cost (0.05 when no call was priced). Thresholds above `bandUpper` throw a `RangeError`.
- `bestThreshold({ ... })`: lowest `totalCost`, the lowest threshold wins a tie.
- `items` (one `EvaluationItem` per text) and `toDict()`, which uses the Python SDK's snake_case keys.

Unknown cost stays `null` and is never counted as 0: a debate whose calls reported no cost has `juryCostUsd: null` and adds to `unpricedCalls`. The default `LiteLLMClient` never reports cost, so debate costs are `null` unless your `llmClient` fills `costUsd`. `examples/typescript/evaluate_jury.ts` runs offline with a stub LLM client.

### Threshold Calibration

`new ThresholdCalibrator(jury)` then `await calibrate({ texts, labels, errorCost=10, escalationCost?, thresholds?, useJury=false })`.

Each text is classified once. An item escalates at threshold t when its confidence is below t or not a finite number, as in `Jury`.

- Without `useJury` the jury never runs: each escalation costs `escalationCost` (default 0.05) and counts as neither right nor wrong, so `accuracy` is the primary accuracy on the items it keeps.
- `useJury: true` runs `JuryEvaluator` with `bandUpper` set to the highest threshold and picks the threshold from the measured sweep. `escalationCost` defaults to the measured mean debate cost, rows also carry `systemAccuracy` and `juryAccuracy`, and the evaluation is kept on `calibrator.evaluationReport`.

Report: `calibrationReport()` returns `bestThreshold`, `useJury` and rows with threshold, accuracy, escalation rate, and total cost (plus `summary` after a jury calibration). `calibrate(...)` mutates `jury.threshold` to the best threshold.

### LLM Transport (`LiteLLMClient`)

- `new LiteLLMClient({ baseUrl?, apiKey?, timeoutMs?, maxAttempts?, logger? })`
- `timeoutMs` (default 60000) applies to each attempt; `maxAttempts` (default 3) counts the first try
- Falls back to env vars: `LITELLM_BASE_URL`, `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`); `LITELLM_API_KEY`, `OPENAI_API_KEY`
- Sends `POST /chat/completions`
- Retries network errors, timeouts, 429 and 5xx responses, waiting for `Retry-After` when the provider sends it (capped at 60 s) and backing off exponentially otherwise; each retry is logged through `logger`
- Returns `{ content, tokens, costUsd }`. `costUsd` is always `undefined` (no viable npm cost-estimation library); a custom `llmClient` can report it
- Throws `No API key configured. Set LITELLM_API_KEY or OPENAI_API_KEY, or inject a custom llmClient.` before sending a request when there is no key

Temperature is automatically omitted for reasoning models (`gpt-5*`, `o1*`, `o3*`, with or without a provider prefix).

### Response Cache (`CachingLLMClient`)

Opt-in LRU wrapper around any `LLMClient`. Keyed on
`(model, systemPrompt, prompt, temperature, responseFormat)`.
Successful responses only; rejections propagate without being cached.
A hit reports `costUsd: 0` and `cached: true`, so it adds nothing to
verdict totals or the cost cap.

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
a shared cache. Caches at any temperature; if you need fresh
stochastic samples, don't wrap.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error: No API key configured. Set LITELLM_API_KEY or OPENAI_API_KEY, or inject a custom llmClient.` (thrown before any HTTP request) | `OPENAI_API_KEY` and `LITELLM_API_KEY` both unset, and no `apiKey` passed | `export OPENAI_API_KEY=...` or `new LiteLLMClient({ apiKey: "..." })` |
| `Error: LLM request failed (401): ...` | API key present but wrong, or wrong provider for the model | Verify the key against the provider; check `OPENAI_BASE_URL` / `LITELLM_BASE_URL` |
| `AbortError: This operation was aborted` after a long wait | Each attempt times out after `timeoutMs` (default 60000), and timeouts are retried up to `maxAttempts` | `new LiteLLMClient({ timeoutMs: 30_000 })`; lower `maxAttempts` to fail sooner |
| `RangeError: confidenceThreshold must be a finite number in [0, 1]` | `confidenceThreshold` outside [0, 1], `NaN` or not a number | Pass a threshold between 0 and 1 |
| `Error: LLMClassifier requires at least one non-empty label.` | `new LLMClassifier({ labels: [] })` or labels that are all blank | Pass at least one label |
| Primary result has `confidence` 0 and the item always escalates | `LLMClassifier` could not use the model's reply; `primaryResult.rawOutput.error` is `invalid_json`, `label_not_in_labels` or `invalid_confidence` | Use a model that honours `response_format` JSON schemas; or wrap your own call in `FunctionClassifier` to control parsing |
| `Error: LLM request failed (429): ...` after retries | Rate-limit budget exhausted after `maxAttempts` (default 3) attempts; the client already waits for `Retry-After`, up to 60 s | Lower `debateConcurrency` and batch `concurrency`; raise `maxAttempts`; use a higher-tier key |
| `verdict.judgeStrategy === "cost_guard_pre_flight"` (no debate ran) | `estimatedMaxDebateCostUsd` exceeded `maxDebateCostUsd`. The estimate counts the summariser and LLM judge calls too, so a cap tuned for 0.2.0 or earlier can now trip | Raise the cap, lower `maxRounds`, lower `estimatedCostPerPersonaUsd` if your calls cost less, or accept the primary classifier verdict |
| `verdict.judgeStrategy === "cost_guard_primary_fallback"` (debate ran partially) | Spend during the debate hit the cap. With the default client no call reports a cost, so every call is charged at `estimatedCostPerPersonaUsd` | Same as above; can still overshoot by up to one concurrency batch (in-flight calls aren't cancellable) |
| `verdict.judgeStrategy === "cost_guard_user_override"` | Your `onCostEstimate` callback returned `false` | Working as intended: the debate was skipped per your policy |
| `verdict.judgeStrategy === "llm_judge_fallback_error"` | The LLM judge call threw (after retries) | The verdict is a majority vote over the final round. Check the judge model and key; consider routing these to review |
| `verdict.judgeStrategy` is `llm_judge_fallback_invalid_json`, `_invalid_label` or `_invalid_confidence` | The judge replied with unparseable JSON, a label outside your labels, or a non-numeric confidence | Same majority-vote fallback. Use a judge model that honours `response_format` |
| `verdict.judgeStrategy === "llm_judge_fallback_personas_failed"` | Every persona call failed, so the judge had nothing to weigh | The primary classifier result is returned; see `debateDegraded` below |
| `verdict.totalCostUsd` is `null` after a debate | The default `LiteLLMClient` cannot estimate cost (no npm pricing library) | Inject a custom `llmClient` that fills `costUsd`; the SDK forwards it through |
| `verdict.totalCostUsd` looks too low | Some calls reported no cost; `verdict.debateTranscript.unpricedCalls` counts them and the total is a lower bound | Make your custom `llmClient` report `costUsd` for every call |
| `classifyBatch` returns fewer / duplicated results than inputs | This was B1, fixed in 0.1.1 | Upgrade; if you still see it, file a bug with a repro |
| Verdict is never escalated even at very low confidence | `personas: []` silently disables escalation (by design) | Pass at least one persona |
| `verdict.debateDegraded` is `true` | One or more persona calls failed (auth, rate-limit exhaustion, unparseable output, a label outside your labels). Failed personas carry no vote; if the whole final round failed, judges return the primary classifier result | Inspect `verdict.personaFailures` and the transcript's `failed` responses; consider routing degraded verdicts to human review |
| One persona's responses have `failed: true` in every round | That persona's `model` is invalid or not available to your key. Its placeholders stay in the transcript and carry no vote | Pass a `logger` (e.g. `new Jury({ ..., logger: console })`) to see why; fix the persona's `model` or remove the persona |
| Debate summary is `undefined` in deliberation mode | The debate stopped early (unanimous labels or `earlyStopMinConfidence`), or the summariser call failed | Nothing to fix for an early stop; otherwise pass a `logger` to see the warning |
| Logs are silent in production | Default logger is `NOOP_LOGGER` (parity with Python being opt-in) | `new Jury({ ..., logger: console })` or pass any object matching the `Logger` interface |

Known problems and their status are tracked in [docs/REVIEW.md](https://github.com/mokhld/llm-jury/blob/main/docs/REVIEW.md).

## Examples

Runnable examples live in `examples/typescript/` at the repo root (require `OPENAI_API_KEY` except `threshold_calibration`):

```bash
node --experimental-strip-types examples/typescript/content_moderation.ts
node --experimental-strip-types examples/typescript/custom_personas.ts
node --experimental-strip-types examples/typescript/legal_compliance.ts
node --experimental-strip-types examples/typescript/threshold_calibration.ts
```

Each example imports from `@llm-jury/core` to mirror real user code. Run `npm install @llm-jury/core` in your project first, or `npm link` the local package when running directly from a clone.

## Testing

```bash
npm test
```

### Real API Smoke Test

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" node --test --experimental-strip-types tests/smoke/real-api.test.ts
```

## CLI

The CLI is for batch workflows. The primary interface is the TypeScript API above. Installing the package
adds an `llm-jury` command (`npx llm-jury --help`, `npx llm-jury --version`). From a clone, build first and run
`node dist/cli/bin.js` instead.

```bash
npx llm-jury classify \
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

With `--classifier function` (the default) the CLI replays predictions stored in the input, so every row needs `text`, `predicted_label` and `predicted_confidence` (0 to 1); the ground-truth `label` field is never read as a prediction. Output rows use the same snake_case keys as the Python CLI. Bad usage exits with code 2 before any LLM call: rows without predictions, out-of-range option values, an unknown option or `--debate-mode`. `classify` writes a `{"text", "error"}` row for each input that failed and exits with code 1 only when every row failed.

Calibration (without `--use-jury` the jury never runs):

```bash
node dist/cli/main.js calibrate \
  --input calibration.jsonl \
  --classifier function \
  --labels safe,unsafe

# Calibrate on measured jury outcomes (makes LLM calls)
node dist/cli/main.js calibrate \
  --input calibration.jsonl \
  --use-jury \
  --personas content_moderation \
  --judge majority \
  --persona-model gpt-5-mini \
  --labels safe,unsafe
```

`calibrate` options: `--error-cost` (default 10), `--escalation-cost` (default 0.05, or the measured mean debate cost with `--use-jury`), `--initial-threshold` (default 0.7), `--use-jury`. Without `--use-jury` the jury options have no effect, and `calibrate` says so on stderr when you pass any.

Evaluation (measure the jury against the primary classifier):

```bash
node dist/cli/main.js eval \
  --input calibration.jsonl \
  --personas content_moderation \
  --judge majority \
  --labels safe,unsafe \
  --max-escalations 200 \
  --output report.json
```

`eval` options: `--output` (full report with per-row results, as JSON), `--band-upper` (default 0.95), `--max-escalations`, `--thresholds` (comma-separated, each at most `--band-upper`), `--error-cost` (default 10), `--escalation-cost` (default: measured mean debate cost), `--concurrency` (default 5), plus the classifier, persona, judge and debate options of `classify`. It prints one JSON line with `best_threshold`, `summary` and `sweep`, the same shape as the Python CLI.

Supported classifier specs: `function`, `llm:<model>`, `huggingface:<model>`.

## License

MIT
