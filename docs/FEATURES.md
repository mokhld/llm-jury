# llm-jury feature backlog

Agent-ready feature briefs. Each item is self-contained: context, API sketch for both
SDKs, files, acceptance criteria. The ranking and reasoning behind the order are in
[REVIEW.md](REVIEW.md) (Part B).

## How to use this file (instructions for an agent)

You were pointed here to implement a feature. Do this:

1. Read `CLAUDE.md` at the repo root first (layout, commands, conventions).
2. Pick the item you were asked for. If none was named, pick the first item in the table
   below whose status is `open` and whose dependencies are all `done`.
3. Set its status to `in progress` in both the table and the item heading.
4. Implement Python first (`packages/python`, the source of truth), then TypeScript
   (`packages/typescript`) with the same behaviour and camelCase names. Both SDKs change
   in the same piece of work unless the brief says the item is single-SDK.
5. Add tests in both SDKs covering every acceptance criterion. Run the full suites and
   linters (commands in `CLAUDE.md`). No real LLM calls in tests; use the fake clients in
   `packages/python/tests/helpers.py` and `packages/typescript/tests/helpers.ts`.
6. Document it: the relevant section of the root `README.md` and the package READMEs, plus
   an entry under `## [Unreleased]` in `CHANGELOG.md`. Keep docs free of em dashes.
7. Set the status to `done (YYYY-MM-DD)` and add a one-line "Shipped:" note under the item
   saying what landed and anything deferred. Update the matching row in `REVIEW.md`
   Part B.
8. Do not commit, push or open a PR unless the person asked you to.

If a brief conflicts with the code (it was written on 2026-09-24), trust the code, adapt
the brief, and note the deviation in the "Shipped:" line.

## Backlog

| ID | Feature | Effort | Depends on | Status |
|---|---|---|---|---|
| FEAT-01 | Jury evaluation harness and outcome-aware calibration | medium-large | BUG-01..04, CLI-01 | done (2026-09-24) |
| FEAT-02 | Human-review routing policy | small | BUG-01, BUG-03 | open |
| FEAT-03 | Score distributions, margin routing, better LLMClassifier confidence | medium | FEAT-01 (to measure it) | open |
| FEAT-04 | Budgets that bind: TS pricing, auto cost estimate, per-debate time budget | medium | BUG-04, BUG-08 | open |
| FEAT-05 | Observability hooks and dollar-based stats | medium | BUG-04 | open |
| FEAT-06 | Streaming, resumable batch and CLI | medium | CLI-01..05 | open |
| FEAT-07 | Reproducible verdict provenance | small | BUG-10 | open |
| FEAT-08 | Input size limits (denial-of-wallet guard) | small | none | open |

BUG/CLI IDs refer to `REVIEW.md` Part A; check their status there before starting.

---

## FEAT-01: Jury evaluation harness and outcome-aware calibration

Status: done (2026-09-24)

Shipped: `JuryEvaluator` / `EvaluationReport` / `EvaluationItem`, `Jury.escalate`,
`calibrate(use_jury=...)`, CLI `eval` and `calibrate --use-jury`, and offline examples,
in both SDKs with a shared fixture. Deviations: the spend guard raises
`TooManyEscalationsError` (a `ValueError`, TS `RangeError`) so the CLI can tell it apart;
`Jury.escalate` raises when the jury has no personas; an item's debate cost is the
verdict total minus the primary cost, so it is unknown when the primary reports no cost;
`summary()` also carries `band_upper` and items also carry primary cost, `jury_degraded`
and `unpriced_calls`; `best_threshold` also takes `thresholds`; `eval` also takes
`--concurrency` (default 5); calibrator `escalation_cost` defaults to None (0.05 without
the jury, the measured mean with it), non-finite confidences escalate there as in `Jury`,
and `calibration_report()` adds `use_jury` (and `summary` in jury mode); the calibrate
stderr note fires only when a jury flag was passed; both CLI `main` functions take an
optional LLM client so tests run offline. Deferred: README public-export and example
lists (left to the parallel docs change); re-measuring DOC-04 latency and cost needs
live runs.

**Problem.** Nobody can show that the jury beats their primary classifier on their data.
`ThresholdCalibrator` never runs the jury: it treats every escalation as a fixed cost with
an implicitly perfect outcome (`calibration/optimizer.py:58-72`). The TS calibrator also
re-classifies each text per threshold and counts escalations as correct
(`optimizer.ts:41-53`). The CLI's jury flags do nothing for `calibrate`. Documented cost
and latency (2-5 s, $0.001-0.005 per escalation) disagree with live runs (27-57 s,
$0.007-0.015).

**API (Python).**
```python
from llm_jury import JuryEvaluator

report = await JuryEvaluator(jury).evaluate(
    texts, labels,
    band_upper=0.95,        # debate every item whose primary confidence < band_upper
    max_escalations=None,   # raise ValueError before spending if more items would be debated
    concurrency=5,
)
report.summary()            # dict: n, primary_accuracy, debated, jury_accuracy_on_debated,
                            # primary_accuracy_on_debated, flips_helped, flips_hurt,
                            # debate_cost_usd (known sum), unpriced_calls, mean_debate_cost_usd,
                            # latency_ms_p50, latency_ms_p95, degraded, fallbacks {strategy: n},
                            # confusion {"primary": {...}, "jury": {...}}
report.threshold_sweep(thresholds=None, error_cost=10.0, escalation_cost=None)
                            # rows: threshold, escalation_rate, system_accuracy,
                            # jury_accuracy, primary_accuracy, errors, total_cost
report.best_threshold(error_cost=10.0, escalation_cost=None)
report.items                # per item: text, expected, primary label/confidence/correct,
                            # jury label/confidence/correct/strategy/cost/duration (if debated)
report.to_dict()
```
- `Jury.escalate(text, primary) -> Verdict`: new public method that runs the escalation
  branch (cost gates, debate, judge) for an existing primary result, bypassing the
  threshold. `Jury.classify` calls it internally, so behaviour is shared.
- The evaluator classifies each text exactly once with `jury.classifier`, then calls
  `jury.escalate` for items with confidence below `band_upper`. It does not touch
  `jury.stats`.
- Sweep maths for threshold t: items with confidence >= t use the primary label, items
  below t use the jury label. `escalation_cost=None` means "use the measured mean debate
  cost", falling back to 0.05 when no call was priced.
  `total_cost = errors * error_cost + escalations * escalation_cost`. Thresholds above
  `band_upper` are rejected because their jury outcomes were not measured.
- `ThresholdCalibrator.calibrate(..., use_jury=False)`: `False` keeps the cheap mode
  (classify once, escalated items excluded from `accuracy`, the Python behaviour). `True`
  runs `JuryEvaluator` and picks the best threshold from the measured sweep. Rows then
  also carry `system_accuracy` and `jury_accuracy`. `accuracy` keeps meaning "primary
  accuracy on non-escalated items" in both modes.

**API (TypeScript).** `new JuryEvaluator(jury).evaluate({ texts, labels, bandUpper,
maxEscalations, concurrency })`, `report.summary()`, `report.thresholdSweep({...})`,
`report.bestThreshold({...})`, `report.items`, `report.toDict()` (snake_case keys, same
as Python), `jury.escalate(text, primary)`, `calibrate({ ..., useJury })`. Fix the TS
calibrator to classify once and exclude escalations from `accuracy`.

**CLI (both).** `llm-jury eval --input labelled.jsonl [classifier, persona, judge, debate
flags as for classify] [--thresholds 0.5,0.6] [--band-upper 0.95] [--max-escalations N]
[--error-cost 10] [--escalation-cost X] [--output report.json]` prints the summary and
the best threshold as JSON. `llm-jury calibrate --use-jury` runs outcome-aware
calibration so the jury flags take effect. Without `--use-jury`, calibrate says on stderr
that jury flags are ignored in cheap mode.

**Files.** New `packages/python/src/llm_jury/evaluation/` and
`packages/typescript/src/evaluation/`; `calibration/optimizer.*`; `jury/core.*`
(`escalate`); `cli/main.*`; package exports; tests; READMEs; CHANGELOG; an offline example
(`examples/evaluate_jury.py` and `.ts`) using a stub LLM client so it runs without an API
key.

**Acceptance criteria.**
- The primary classifier is called exactly once per text (asserted with a counting
  classifier) in both the evaluator and both calibrator modes, in both SDKs.
- The jury runs only for items below `band_upper`; `max_escalations` raises before any
  LLM call when exceeded.
- With a scripted fake jury: `flips_helped` / `flips_hurt`, `system_accuracy` per
  threshold and `best_threshold` match hand-computed values. The fixture is the same in
  both SDKs and gives identical numbers.
- An always-wrong jury makes `best_threshold` pick the lowest threshold; an always-right
  jury with a cheap escalation cost picks the highest.
- Unknown debate cost is reported as `None`/`null` with `unpriced_calls > 0`, never 0.
- CLI `eval` and `calibrate --use-jury` work end to end with an injected fake client.

---

## FEAT-02: Human-review routing policy

Status: open

**Problem.** The main user story is "resolve what is safe to resolve automatically, send
the rest to a human". Today users string-match several `judge_strategy` fallback markers
(`cost_guard_*`, `llm_judge_fallback_*`, `primary_classifier`), and a 2-1 split vote
comes back looking like any other verdict. `debate_degraded` only covers persona
failures.

**API (Python).**
```python
from llm_jury import ReviewPolicy
jury = Jury(..., review_policy=ReviewPolicy(
    min_confidence=0.75,       # verdict confidence below this -> "low_confidence"
    max_dissent=0.34,          # share of valid final-round votes against the winner above this -> "split_jury"
    on_degraded=True,          # persona_failures > 0 -> "degraded"
    on_fallback=True,          # judge_strategy starts with "llm_judge_fallback" -> "judge_fallback"
    on_cost_guard=True,        # judge_strategy starts with "cost_guard" -> "cost_guard"
    on_overturn_below=None,    # jury label != primary label and confidence below this -> "overturn_low_confidence"
), on_review=callback)
verdict.needs_review      # bool
verdict.review_reasons    # list[str], stable reason codes above
verdict.resolution        # "fast_path" | "jury" | "fallback_primary" | "fallback_vote"
```
Without a policy, `needs_review` is False and `review_reasons` is empty, but `resolution`
is always set. Jury sets these fields after judging (authoritative, like
`was_escalated`) and calls `on_review` once for each verdict that needs review. All three
fields are serialised. TS: `reviewPolicy`, `onReview`, `needsReview`, `reviewReasons`,
`resolution`.

**Files.** New `review.py` / `review.ts`; `judges/base.*` (Verdict fields);
`jury/core.*`; exports; CLI `--review-min-confidence` and `--review-max-dissent` flags
(optional); tests; docs.

**Acceptance criteria.** One test per reason code in each SDK; `resolution` is correct for
the fast path, a judged verdict, each cost guard, and each judge fallback; `on_review`
fires exactly once per flagged verdict; README troubleshooting table points to
`review_reasons` instead of raw strategy strings.

---

## FEAT-03: Score distributions, margin routing, better LLMClassifier confidence

Status: open

**Problem.** Routing is only as good as the confidence behind it. The paper routes
multi-class items on confidence and on the top-1/top-2 margin; llm-jury uses confidence
only (`.content/value-analysis.md`, "What we don't implement"). LLM self-reported
confidence runs high: in the Feb 2026 live runs, inputs written to escalate came back
`was_escalated: false` at threshold 0.85. The sklearn and HF adapters already hold the
full distribution in `raw_output` but throw it away.

**API.**
- `ClassificationResult.scores: dict[str, float] | None` plus a `margin` property (top1 -
  top2, None with fewer than two scores). Populated by the sklearn and HF adapters, and by
  `FunctionClassifier` when `fn` returns `(label, confidence, scores)`.
- `Jury(margin_threshold=None)`: escalate when `confidence < threshold` or
  `margin < margin_threshold`.
- `LLMClassifier(confidence_source="self_reported" | "logprobs")`. `"logprobs"` asks the
  client for token logprobs (the `LLMClient.complete` protocol gains optional
  `logprobs: bool` and `top_logprobs: int` that `LiteLLMClient` passes through and returns
  as `payload["logprobs"]`). Confidence is the probability of the chosen label's first
  token at the label position; fall back to self-reported when logprobs are missing.
  `CachingLLMClient`'s key must include the new arguments.
- Calibrator and evaluator sweep `margin_threshold` too (FEAT-01).
- TS: `scores?: Record<string, number>`, `marginThreshold`, `confidenceSource`.

**Acceptance criteria.** Margin escalation tests for binary and 3-class inputs; adapters
populate `scores` in both SDKs; logprob confidence computed from a fixture payload;
fallback path covered; evaluator reports a margin sweep.

---

## FEAT-04: Budgets that bind

Status: open

**Problem.** After BUG-04, unknown costs are no longer reported as 0 and unpriced calls
are charged at the flat per-call estimate, but TS still has no way to price calls, the
flat estimate ($0.01 per call) is a guess, and debates can take 30 to 60 seconds with no
overall deadline.

**API.**
- `LiteLLMClient(pricing={"gpt-5-mini": {"input_per_1m": 0.25, "output_per_1m": 2.0}})`
  in both SDKs (TS `pricing`, `inputPer1M`, `outputPer1M`). TS computes `costUsd` from
  `usage.prompt_tokens` / `completion_tokens`; Python uses the table before litellm's own.
  Do not ship a built-in price table (it goes stale); document the shape.
- `Jury(estimated_cost_per_persona_usd="auto")`: running mean of observed priced calls,
  starting from 0.01 until 5 priced calls have been seen.
- `Jury(max_debate_seconds=None)` / `maxDebateMs`: when the deadline passes, start no new
  round, judge the rounds already completed with the configured judge, and add
  `"deadline"` to `review_reasons` if FEAT-02 exists (otherwise set
  `judge_strategy` suffix `_deadline_partial`). TS passes an `AbortSignal` to the client.
- `JuryStats`: `total_cost_usd`, `debate_cost_usd`, `unpriced_calls`, and
  `estimated_savings_usd` (fast-path count times mean debate cost).

**Acceptance criteria.** TS cost computed from a usage fixture; auto estimate converges in
a test; deadline test with a slow fake client proves no new round starts after the
deadline and the verdict is still produced; stats fields covered in both SDKs.

---

## FEAT-05: Observability hooks and dollar-based stats

Status: open

**Problem.** The spec promised "log every debate, every persona response, every judge
decision" (`../llm-jury-spec.md` objective 9). Today only warnings are logged, there is no
per-call latency, and `JuryStats.cost_savings_vs_always_escalate` is a count ratio.

**API.**
- `Jury(on_event=callable)` / `onEvent`. Event:
  `LLMCallEvent(stage: "primary"|"persona"|"summary"|"judge", persona: str | None,
  round: int | None, model, tokens, cost_usd, latency_ms, cached: bool, error: str | None)`.
  Emitted once per LLM call, including failures. Exceptions raised by the callback are
  logged and swallowed.
- `PersonaResponse.latency_ms` and `.model`.
- `JuryStats`: `degraded`, `fallbacks` (strategy -> count), `overturned` (jury label
  differs from primary) and `overturn_rate`. Overturn rate is a production KPI that needs
  no ground truth.
- Optional OpenTelemetry adapter as a Python extra (`llm-jury-classifier[otel]`,
  `llm_jury.integrations.otel`) and a TS subpath export, creating one span per verdict
  and one child span per LLM call. Ship it as a second step if time is short.

**Acceptance criteria.** Event count and fields asserted for each debate mode in both
SDKs; callback exceptions do not break `classify`; stats fields covered.

---

## FEAT-06: Streaming, resumable batch and CLI

Status: open

**Problem.** `classify_batch` returns only when everything is done, and the CLI reads the
whole JSONL into memory and writes only at the end. At measured costs a 10k-row job with
25% escalation is about $27 and hours of wall time, and a crash loses all of it. The CLI
cannot load custom personas. `Classifier.classify_batch` exists but the Jury never calls
it, so vectorised classifiers are called one text at a time.

**API.**
- `async for index, result in jury.classify_stream(texts, concurrency=10,
  return_exceptions=True)` yields results as they finish. TS: `for await (const { index,
  result } of jury.classifyStream(texts, { concurrency }))`.
- Optional `primary_batch_size`: run the primary pass through `classifier.classify_batch`
  in chunks before escalating.
- CLI `classify`: stream the input, append each output row as it completes and flush,
  carry the row's `id` (or its line number) into the output, `--resume` skips ids already
  present in the output file, `--personas-file personas.json` (list of persona objects,
  same shape as `PersonaRegistry.custom`), `--progress` to stderr.
- CLI defaults are `independent` mode and 1 round while the SDK defaults to deliberation
  with 2 rounds. Ask the maintainer before changing CLI defaults (it changes cost); if
  unchanged, print the effective mode on stderr.

**Acceptance criteria.** Stream yields in completion order with correct indexes; `--resume`
test with a partially written output file skips done rows; personas file validated with
a clear error; interrupting a run leaves every completed row on disk.

---

## FEAT-07: Reproducible verdict provenance

Status: open

**Problem.** Compliance users need to reproduce a decision. A verdict records
`library_version` and `created_at`, but not the threshold, debate config, persona
prompts or judge prompt that produced it.

**API.** `Verdict.provenance: dict` set by Jury on every verdict:
`threshold`, `debate_config` (mode, max_rounds, include flags, early stop),
`personas` (name, model, temperature, `sha256` of system_prompt), `judge` (class name,
model and system-prompt hash when present), `classifier` (class name, model when
present), `prompt_template_version` (a constant bumped whenever prompt builders change),
`library_version`. Serialised by `to_dict` / `toDict`. TS mirrors with camelCase keys in
the object and snake_case in `toDict`.

**Acceptance criteria.** Hashes are stable across runs and identical between the SDKs for
the built-in persona sets (same prompt text, same hash); every return path sets
`provenance`; a test fails if a prompt builder changes without bumping
`prompt_template_version` (snapshot of the rendered template for a fixed input).

---

## FEAT-08: Input size limits

Status: open

**Problem.** Old audit item S4 (denial of wallet): SECURITY.md tells callers to cap input
length themselves, and the pre-flight estimate ignores input size, so a 100k-token input
passes the same guard as a tweet.

**API.** `Jury(max_input_chars=None, on_oversized="skip_debate")` with
`on_oversized` in `"skip_debate" | "truncate" | "raise"`. `skip_debate` returns the primary
result with `judge_strategy="input_too_large"`; `truncate` debates the first
`max_input_chars` characters and records `input_truncated=True` on the transcript;
`raise` raises `ValueError` / `RangeError` before any LLM call. TS: `maxInputChars`,
`onOversized`.

**Acceptance criteria.** One test per mode in each SDK; no LLM call happens for `raise`
and `skip_debate`; SECURITY.md updated to point at the option.
