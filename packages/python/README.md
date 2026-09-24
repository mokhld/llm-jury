# llm-jury

**When your classifier is uncertain, let a configurable jury of LLM personas debate and return an auditable verdict.**

[![PyPI](https://img.shields.io/pypi/v/llm-jury-classifier)](https://pypi.org/project/llm-jury-classifier/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/mokhld/llm-jury/blob/main/LICENSE)

## Overview

`llm-jury` is an SDK, not a hosted API. Your app imports it directly:

```python
from llm_jury import Jury, PersonaRegistry
```

The PyPI package is `llm-jury-classifier`. Do not `pip install llm-jury`: that name belongs to an unrelated project that also installs an `llm_jury` module.

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
pip install llm-jury-classifier
```

Optional extras:

```bash
pip install "llm-jury-classifier[sklearn]"
pip install "llm-jury-classifier[huggingface]"
pip install "llm-jury-classifier[all]"
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

`jury.classify(text)` returns a `Verdict`. There are two shapes depending on whether the input was escalated. The JSON below is what `verdict.to_dict()` returns. The samples are illustrative and shortened (persona `raw_response` is left out). The durations and costs are made up; real escalations take far longer and cost more (see [Important Notes](#important-notes) for measured numbers).

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

### Escalated (confidence below threshold)

When confidence is too low, the input goes through persona debate and a judge produces the final verdict. In this sample the three personas agree in the opening round, so the debate stops there with no second round and no summary.

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

### Verdict field reference

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Final classification |
| `confidence` | `float` | Final confidence (0.0 to 1.0) |
| `reasoning` | `str` | Human-readable explanation |
| `was_escalated` | `bool` | Whether debate was triggered |
| `primary_result` | `ClassificationResult` | Fast-path classifier output |
| `debate_transcript` | `DebateTranscript \| None` | Full debate audit trail (see below); None if no debate ran |
| `judge_strategy` | `str` | Strategy that produced the verdict, including the fallback markers listed under [Troubleshooting](#troubleshooting) |
| `total_duration_ms` | `int` | Wall-clock time (ms) |
| `total_cost_usd` | `float \| None` | Primary classifier plus debate and judge cost in USD. None when any part is unknown; a lower bound when `debate_transcript.unpriced_calls > 0` |
| `persona_failures` | `int` | Persona calls across the debate that failed (LLM error, unparseable output, or a label outside the configured set) |
| `debate_degraded` | `bool` | Property (also in `to_dict()`): True when `persona_failures > 0`, so the verdict was decided by fewer jurors than configured. Useful for routing to human review |
| `judge_details` | `dict \| None` | `LLMJudge` only: `key_agreements`, `key_disagreements`, `decisive_factor`. None for other judges and fallbacks |
| `library_version` | `str` | `llm-jury-classifier` version that produced the verdict |
| `created_at` | `str` | ISO 8601 UTC timestamp |

### Debate transcript fields

| Field | Type | Description |
|---|---|---|
| `input_text` | `str` | The text that was classified |
| `primary_result` | `ClassificationResult` | Primary classifier output |
| `rounds` | `list[list[PersonaResponse]]` | One list per round, in order |
| `summary` | `str \| None` | Summariser output in deliberation mode. None when the debate stopped early, in other modes, or when the summariser call failed |
| `duration_ms` | `int` | Debate wall-clock time (ms) |
| `total_tokens` | `int` | Tokens used by persona and summariser calls |
| `total_cost_usd` | `float \| None` | Sum of the persona and summariser calls that reported a cost; None when none did |
| `unpriced_calls` | `int` | Debate calls that reported no cost. Non-zero means `total_cost_usd` is a lower bound |
| `persona_biases` | `dict[str, str]` | Persona name to `known_bias`, for personas that declare one. The LLM judge sees these |
| `persona_failures` | `int` | Property: failed persona responses across all rounds |

### Persona response fields

| Field | Type | Description |
|---|---|---|
| `persona_name` | `str` | Which persona |
| `label` | `str` | This persona's classification |
| `confidence` | `float` | This persona's confidence |
| `reasoning` | `str` | Full reasoning chain |
| `key_factors` | `list[str]` | Key decision factors |
| `dissent_notes` | `str \| None` | Rebuttal in deliberation/adversarial modes |
| `raw_response` | `str \| None` | The model's raw reply |
| `tokens_used` | `int` | Tokens consumed |
| `cost_usd` | `float \| None` | API cost for this call; None when the client reported none |
| `failed` | `bool` | True when this response is a placeholder for a failed persona call. Failed responses stay in the transcript for audit but carry no vote |

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

- **Temperature is handled automatically.** The SDK omits the temperature parameter for reasoning models (`gpt-5*`, `o1*`, `o3*`, also behind a provider prefix such as `openai/gpt-5-mini`). No configuration needed.
- **Escalation is strictly `< threshold`**: confidence exactly equal to the threshold does NOT escalate. A missing or non-finite primary confidence (`None`, NaN) always escalates.
- **Automatic retry**: each LLM call gets 3 attempts in total (the first try plus 2 retries, via tenacity) on connection errors, timeouts, 429 and 5xx responses; other errors fail at once. The client waits for the provider's `Retry-After` header when it sends one (capped at 60 s) and backs off exponentially otherwise. Change the count with `LiteLLMClient(max_attempts=...)`.
- **Default debate mode is deliberation**, the full 4-stage CEJ pipeline. For cheaper and faster runs use `DebateConfig(mode=DebateMode.INDEPENDENT)`.
- **Deliberation stops early** after any round, the opening one included, when the personas' labels are unanimous or `early_stop_min_confidence` is met. A debate that stops early has no summary.
- **Latency and cost of an escalation**: a debate makes several rounds of LLM calls. Live runs in February 2026 with the default `gpt-5-mini`, three personas and a mix of debate modes and judges took 27 to 57 s and cost $0.007 to $0.015 per escalated item. Use that as a rough guide only; your models, personas and inputs will change it. The fast path costs one primary classifier call.
- **Cost tracking**: each call's cost is estimated from litellm's model pricing table (not from provider billing), so a model litellm does not price reports `None`. Unknown cost is `None`, never 0. An escalated `total_cost_usd` includes the primary classifier, is `None` when the primary cost or the whole debate cost is unknown, and is a lower bound when `debate_transcript.unpriced_calls > 0`.
- **The cost cap is checked twice.** Before a debate, `estimated_max_debate_cost_usd` is compared with `max_debate_cost_usd`; the estimate counts every persona call in every round, the summariser in deliberation mode and the judge when it is an `LLMJudge`, each at `estimated_cost_per_persona_usd` (default $0.01 per call). During the debate, reported spend is compared with the cap, and calls that reported no cost are charged at that same per-call estimate.
- **Empty personas disables escalation**: If you pass `personas=[]`, the jury always returns the primary classifier result.
- **Untrusted input**: every prompt fences the input in `<input>` tags marked as untrusted data, and labels returned by models are checked against your labels. See [Prompt injection and untrusted input](https://github.com/mokhld/llm-jury#prompt-injection-and-untrusted-input) for what callers should still do.

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
| `estimated_cost_per_persona_usd` | `0.01` | Estimated cost of one LLM call (persona, summariser or judge). Used for the pre-flight estimate and charged against the cap for calls that report no cost |
| `debate_concurrency` | `5` | Max concurrent persona calls |
| `on_escalation` | `None` | Fires when input is escalated to debate. `(text, primary_result) -> None` |
| `on_cost_estimate` | `None` | Fires with `(estimated_max_debate_cost_usd, text)` immediately before a debate would run. Return `False` to skip the debate (verdict marked `cost_guard_user_override`); return `True` / `None` to proceed. |
| `on_verdict` | `None` | Fires once with every verdict `classify` returns, including fast-path and cost-guard verdicts. `(verdict) -> None` |
| `llm_client` | `None` | LLM transport override |
| `logger` | `None` (uses `logging.getLogger(__name__)`) | Logger override |

Methods:

- `await classify(text)`: classify a single input
- `await classify_batch(texts, concurrency=10, return_exceptions=False)`: classify multiple inputs. With `return_exceptions=True`, a failing text yields its exception in-slot instead of rejecting the whole batch. Without it, the first failure raises and no further text starts a debate.

Behavior notes:

- Escalation condition is strictly `< threshold` (exactly equal does not escalate).
- If `personas` is empty, jury escalation is effectively disabled.
- Failed persona calls (LLM error, unparseable output, or a label outside the configured set) are kept in the transcript as placeholders with `failed=True` but carry no vote. If the whole final round failed, judges return the primary classifier result. `Verdict.persona_failures` counts them and `Verdict.debate_degraded` is True when any persona failed; use it to route degraded verdicts to human review.
- `Jury.estimated_max_debate_cost_usd` (property) is the pre-flight estimate: (persona calls + 1 summariser call in deliberation mode + 1 judge call for an `LLMJudge`) x `estimated_cost_per_persona_usd`. Persona calls are `len(personas) x max_rounds` in deliberation mode and `len(personas)` in the other modes. If it exceeds `max_debate_cost_usd`, no debate runs and `judge_strategy` is `cost_guard_pre_flight`.
- If spend during the debate exceeds `max_debate_cost_usd`, the result falls back to the primary classifier with `judge_strategy` set to `cost_guard_primary_fallback`. Calls that reported no cost are charged at `estimated_cost_per_persona_usd`.
- `on_cost_estimate` runs after the escalation decision but before any LLM call for the debate, *and* before the `max_debate_cost_usd` guard. Lets you layer per-tenant budgets, time-of-day gates, etc. on top of the hard cap.

Stats: `jury.stats.total`, `fast_path`, `escalated`, `escalation_rate`, `cost_savings_vs_always_escalate`.

### `DebateConfig` Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `deliberation` | Debate mode |
| `max_rounds` | `2` | Max deliberation rounds |
| `include_primary_result` | `true` | Include primary result in prompts |
| `include_confidence` | `true` | Include confidence in prompt context |
| `early_stop_min_confidence` | `None` | Opt-in early stop for deliberation mode. When set, the debate ends after any round, the opening one included, whose **lowest** persona confidence is `>=` this value, even if personas disagree on label. Unanimous labels end it regardless. None means only unanimous labels stop early. |

### Personas

Persona fields: `name`, `role`, `system_prompt`, `model="gpt-5-mini"`, `temperature=0.3`, `known_bias=None`.

### Classifiers (API)

All classifiers implement `classify(text)` and expose `labels`.

- **FunctionClassifier**: `FunctionClassifier(fn, labels)` where `fn` may be sync or async
- **LLMClassifier**: `LLMClassifier(model="gpt-5-mini", labels=None, system_prompt=None, llm_client=None, temperature=0.0)`. `labels` must hold at least one label. Sends a JSON schema with the labels as an enum; the returned label is matched to yours (exact, then case-insensitive). Output it cannot use returns confidence 0 (so the jury escalates it) with the reason in `raw_output["error"]`: `invalid_json`, `label_not_in_labels` or `invalid_confidence`.
- **SklearnClassifier**: `SklearnClassifier(model, labels, vectorizer=None)` uses `predict_proba`. Columns are named by `model.classes_` when those are the same set as `labels`, otherwise by position; a label count that differs from `classes_` raises `ValueError`. Inference runs in a worker thread.
- **HuggingFaceClassifier**: `HuggingFaceClassifier(model_name, device="cpu", labels=None)` (requires `transformers`). Without `labels`, the label list comes from the model's full score list on the first call. Inference runs in a worker thread.

### Judge Strategies (API)

- **MajorityVoteJudge**: `MajorityVoteJudge()`. Confidence is the fraction of the final round's valid responses voting for the winning label.
- **WeightedVoteJudge**: `WeightedVoteJudge()`. Confidence comes from confidence-weighted label scores.
- **LLMJudge**: `LLMJudge(model="gpt-5-mini", system_prompt=None, temperature=0.0, llm_client=None)`. Reads every round, the summary and each persona's `known_bias`, and answers with a label-enum JSON schema. On success `judge_strategy` is `llm_judge` and `verdict.judge_details` holds `key_agreements`, `key_disagreements` and `decisive_factor`. When its call fails or its output is unusable, it returns a majority vote over the final round (`llm_judge_fallback_error`, `llm_judge_fallback_invalid_json`, `llm_judge_fallback_invalid_label`, `llm_judge_fallback_invalid_confidence`), or the primary result if that round has no valid responses. When every persona failed it skips its call and returns the primary result (`llm_judge_fallback_personas_failed`).
- **BayesianJudge**: `BayesianJudge(persona_priors=None)`. Uses persona priors/reliability maps if provided.

### Threshold Calibration

`ThresholdCalibrator(jury)` then `await calibrate(texts, labels, error_cost=10.0, escalation_cost=0.05, thresholds=None)`.

Report: `calibration_report()` returns rows with threshold, accuracy, escalation rate, and total cost. `calibrate(...)` mutates `jury.threshold` to the best threshold.

### LLM Transport (`LiteLLMClient`)

```python
from llm_jury import Jury
from llm_jury.llm import LiteLLMClient

client = LiteLLMClient(
    timeout_seconds=60.0,  # per request; None keeps litellm's default
    max_attempts=3,        # total attempts, the first one included
    api_key=None,          # passed to litellm when set
    api_base=None,         # passed to litellm when set
)
jury = Jury(classifier=..., personas=..., llm_client=client)
```

- `complete(model, system_prompt, prompt, temperature=0.0, response_format=None)` calls `litellm.acompletion`
- Retries connection errors, timeouts, 429 and 5xx responses, waiting for `Retry-After` when the provider sends it (capped at 60 s) and backing off exponentially otherwise
- Returns: `{content, tokens, cost_usd}` (`cost_usd` is `None` when litellm cannot price the model)
- Raises a runtime error if `litellm` is not installed and no custom `llm_client` is injected.

### Response Cache (`CachingLLMClient`)

Opt-in LRU wrapper around any `LLMClient`. Keyed on
`(model, system_prompt, prompt, temperature, response_format)`.
Successful responses only; exceptions propagate without being cached.
A hit reports `cost_usd` 0.0 and `cached: True`, so it adds nothing to
verdict totals or the cost cap.

```python
from llm_jury import CachingLLMClient, Jury
from llm_jury.llm import LiteLLMClient

jury = Jury(
    # ...
    llm_client=CachingLLMClient(
        LiteLLMClient(),
        max_size=1000,        # LRU cap
        ttl_seconds=3600,     # optional; omit for no expiry
    ),
)
```

`hits`, `misses`, and `size` are exposed for introspection. Call
`clear()` to drop everything. The cache is in-process and per-instance;
share the `CachingLLMClient` object across `Jury` instances if you want
a shared cache. Caches at any temperature; if you need fresh stochastic
samples, don't wrap.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Authentication`/`401` error on first LLM call | `OPENAI_API_KEY` not set, or wrong provider key for the model | `export OPENAI_API_KEY=...`; or pass `llm_client=LiteLLMClient(api_key=...)` |
| `ImportError: No module named 'litellm'` | `litellm` not installed and no custom `llm_client` injected | `pip install llm-jury-classifier` (litellm is a hard dep); or inject your own `llm_client` |
| `ImportError: transformers` / `numpy` | Using `HuggingFaceClassifier` / `SklearnClassifier` without the extras | `pip install "llm-jury-classifier[huggingface]"` or `[sklearn]` |
| `ValueError: LLMClassifier requires at least one non-empty label.` | `LLMClassifier(labels=[])` or labels that are all blank | Pass at least one label |
| `ValueError: confidence_threshold must be a finite number in [0, 1]` | `Jury(confidence_threshold=...)` outside [0, 1], NaN or not a number | Pass a threshold between 0 and 1 |
| Primary result has `confidence` 0.0 and the item always escalates | `LLMClassifier` could not use the model's reply; `primary_result.raw_output["error"]` is `invalid_json`, `label_not_in_labels` or `invalid_confidence` | Use a model that honours `response_format` JSON schemas; or wrap your own call in `FunctionClassifier` to control parsing |
| Repeated 429s and the whole call fails | Rate-limit budget exhausted after 3 attempts per call (the client already waits for `Retry-After`, up to 60 s) | Lower `debate_concurrency` and batch `concurrency`; raise `LiteLLMClient(max_attempts=...)`; use a higher-tier key |
| `verdict.judge_strategy == "cost_guard_pre_flight"` (no debate ran) | `estimated_max_debate_cost_usd` exceeded `max_debate_cost_usd`. The estimate counts the summariser and LLM judge calls too, so a cap tuned for 0.2.0 or earlier can now trip | Raise the cap, lower `max_rounds`, lower `estimated_cost_per_persona_usd` if your calls cost less, or accept the primary classifier verdict |
| `verdict.judge_strategy == "cost_guard_primary_fallback"` (debate ran partially) | Spend during the debate hit the cap; calls that reported no cost are charged at `estimated_cost_per_persona_usd` | Same as above; spend can still overshoot by up to one concurrency batch (in-flight calls aren't cancellable) |
| `verdict.judge_strategy == "cost_guard_user_override"` | Your `on_cost_estimate` callback returned `False` | Working as intended: the debate was skipped per your policy |
| `verdict.judge_strategy == "llm_judge_fallback_error"` | The LLM judge call raised (after retries) | The verdict is a majority vote over the final round. Check the judge model and key; consider routing these to review |
| `verdict.judge_strategy` is `llm_judge_fallback_invalid_json`, `_invalid_label` or `_invalid_confidence` | The judge replied with unparseable JSON, a label outside your labels, or a non-numeric confidence | Same majority-vote fallback. Use a judge model that honours `response_format` |
| `verdict.judge_strategy == "llm_judge_fallback_personas_failed"` | Every persona call failed, so the judge had nothing to weigh | The primary classifier result is returned; see `debate_degraded` below |
| `verdict.total_cost_usd is None` | The model is not in litellm's pricing table, or the primary classifier reported no cost | Check `litellm.model_cost`; pin to a known-priced model; or compute cost yourself in a custom `llm_client` |
| `verdict.total_cost_usd` looks too low | Some calls reported no cost; `verdict.debate_transcript.unpriced_calls` counts them and the total is a lower bound | Same as above |
| Verdict is never escalated even at very low confidence | `personas=[]` silently disables escalation (by design) | Pass at least one persona |
| `verdict.debate_degraded` is `True` | One or more persona calls failed (auth, rate-limit exhaustion, unparseable output, a label outside your labels). Failed personas carry no vote; if the whole final round failed, judges return the primary classifier result | Inspect `verdict.persona_failures` and the transcript's `failed` responses; consider routing degraded verdicts to human review |
| One persona's responses have `failed=True` in every round | That persona's `model` is invalid or not available to your key. Its placeholders stay in the transcript and carry no vote | Inspect logs; fix the persona's `model` or remove the persona |
| Debate summary is `None` in deliberation mode | The debate stopped early (unanimous labels or `early_stop_min_confidence`), or the summariser call failed (a warning is logged) | Nothing to fix for an early stop; otherwise verify the first persona's model is reachable |
| A call hangs about 60 s, then fails | `LiteLLMClient` times out each request after 60 s by default and retries timeouts | `LiteLLMClient(timeout_seconds=30)` |
| `verdict.total_duration_ms` is `0` from a custom judge | Custom judge didn't set the field; Jury only backfills when at default | Set `total_duration_ms` in your judge if you want a custom value |

Known problems and their status are tracked in [docs/REVIEW.md](https://github.com/mokhld/llm-jury/blob/main/docs/REVIEW.md).

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

With `--classifier function` (the default) the CLI replays the stored predictions, so every row needs `text`, `predicted_label` and `predicted_confidence` (0 to 1); the ground-truth `label` field is never read as a prediction. Bad usage exits with code 2 before any LLM call: rows without predictions, out-of-range option values, or an unknown `--debate-mode`. `classify` writes a `{"text", "error"}` row for each input that failed and exits with code 1 only when every row failed.

Input JSONL format for `calibrate` (requires ground-truth `label`):
```json
{"text": "some text", "label": "safe", "predicted_label": "safe", "predicted_confidence": 0.95}
```

Supported classifier specs: `function`, `llm:<model>`, `huggingface:<model>`.

## License

MIT
