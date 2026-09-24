# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file covers both SDKs (Python `llm-jury-classifier`, TypeScript
`@llm-jury/core`). Where a change applies to only one SDK it is
marked **[py]** or **[ts]**.

## [Unreleased]

### Added
- `Verdict.judge_details` / `judgeDetails`: the LLM judge's key agreements,
  key disagreements and decisive factor (previously requested and dropped).
- `DebateTranscript.unpriced_calls` / `unpricedCalls` (LLM calls that reported
  no cost) and `DebateTranscript.persona_biases` / `personaBiases` (persona
  `known_bias` values, now shown to the LLM judge as an expert roster).
- **[py]** `LiteLLMClient(timeout_seconds=60.0, max_attempts=3, api_key=None,
  api_base=None)`. **[ts]** `LiteLLMClientOptions.maxAttempts`.
- **[ts]** `HuggingFaceClassifier` accepts a `labels` option.
- New `judge_strategy` markers: `llm_judge_fallback_error`,
  `llm_judge_fallback_invalid_label`, `llm_judge_fallback_invalid_confidence`.

### Changed
- `LLMClassifier` and `LLMJudge` send strict JSON-schema `response_format`
  with the labels as an enum. Custom `LLMClient`s now receive
  `response_format` on classifier and judge calls too.
- Labels from personas, `LLMClassifier` and `LLMJudge` are matched to the
  configured labels (exact, then case-insensitive) and returned in the
  configured spelling.
- `llm_judge_fallback_invalid_json` now returns a majority vote over the final
  round instead of the primary result. All LLM-judge fallbacks return the
  primary result only when the final round has no valid votes.
- Unknown cost is reported as `None` / `null`, never 0. An escalated
  `total_cost_usd` includes the primary classifier's cost and is `None` when
  the primary or the debate part is unknown; it is a lower bound when
  `debate_transcript.unpriced_calls > 0`. `FunctionClassifier`,
  `HuggingFaceClassifier` and `SklearnClassifier` report a cost of 0.
- `estimated_cost_per_persona_usd` now means the estimated cost per LLM call,
  and `estimated_max_debate_cost_usd` counts the summariser and LLM judge
  (8 calls instead of 6 for 3 personas x 2 rounds). Existing
  `max_debate_cost_usd` values may now stop debates at pre-flight.
- Deliberation ends after the opening round when its labels are unanimous or
  `early_stop_min_confidence` is met (no second round, no summary).
- Every prompt wraps the input in `<input>` tags with a note that it is
  untrusted data.
- CLI `--classifier function` requires `predicted_label` and
  `predicted_confidence` on every row; it no longer reads the ground-truth
  `label` as a prediction.
- CLI numeric flags are range-checked and an invalid `--debate-mode` is a
  usage error (exit code 2) in both SDKs.
- **[py]** `HuggingFaceClassifier` and `SklearnClassifier` run inference in a
  worker thread.
- `SklearnClassifier` maps probability columns by the model's `classes_` /
  `classes` when they name the labels, and rejects a label count mismatch.

### Fixed
- Verdicts could carry labels outside the configured set (for example an LLM
  judge answering `"Unsafe - borderline"`).
- A NaN or non-numeric confidence could skip escalation (TS `NaN < threshold`
  is false; Python clamped NaN to 1.0) or crash `LLMClassifier`. Non-finite
  primary confidence now escalates, and `confidence_threshold` must be a
  finite number in [0, 1].
- A failing LLM judge call rejected `classify` and discarded the paid debate.
- The mid-flight `max_debate_cost_usd` guard never tripped when costs were
  unknown (always the case with the TS default client); unpriced calls are
  now charged at the per-call estimate. Float noise at an exact cap no longer
  discards a finished debate.
- Cache hits were billed again in verdict totals and against the cost cap.
- `on_verdict` / `onVerdict` fired only for judged verdicts; it now fires for
  every verdict, including the fast path and cost-guard fallbacks.
- Fail-fast `classify_batch` kept starting debates after it had rejected.
- **[py]** `LiteLLMClient` had no timeout (litellm's default is 6000 s).
- **[ts]** Errors with a 4xx status were retried when their message text
  contained a 5xx-looking number. `Retry-After` is now honoured (both SDKs,
  capped at 60 s).
- Provider-prefixed reasoning models such as `openai/gpt-5-mini` were sent a
  temperature.
- **[ts]** The npm-installed `llm-jury` command did nothing and exited 0
  (the bin is now `dist/cli/bin.js`); `--version` printed `0.1.0`; invalid
  numeric flags hung or queried zero personas; output rows lacked
  `debate_degraded`.
- **[ts]** `HuggingFaceClassifier` fixed its labels to the first call's single
  top result.
- CLI `calibrate` with the `function` classifier reported perfect accuracy by
  reading ground truth as predictions.

## [0.2.0] — 2026-08

### Added
- **Debate health surfaced on `Verdict`** (both SDKs).
  `Verdict.persona_failures` / `personaFailures` counts persona
  calls across the debate that failed (LLM error or unparseable
  output), set authoritatively by `Jury` after judging (like
  `was_escalated`). `Verdict.debate_degraded` / `debateDegraded`
  is true when the count is non-zero — the verdict was decided by
  fewer jurors than configured, or fell back to the primary
  classifier entirely. Both fields are serialised by
  `to_dict()` / `toDict()` so pipelines can route degraded
  verdicts to human review. `PersonaResponse` gains a `failed`
  flag; `DebateTranscript.persona_failures` (py property) and
  `countPersonaFailures(rounds)` (ts helper) expose the raw count.
  `Jury` logs a warning whenever a degraded verdict is produced.
- **Batch error isolation**: `classify_batch(texts, concurrency,
  return_exceptions=False)` / `classifyBatch(texts, concurrency,
  returnExceptions)`. When enabled, a failing text yields its
  exception in-slot (mirroring `asyncio.gather` /
  `Promise.allSettled` semantics) instead of rejecting the whole
  batch and discarding completed verdicts and their spend.
  Defaults preserve the old fail-fast behaviour.
- **[py]** `py.typed` marker — type checkers (mypy, pyright) now
  recognise the package as typed and surface its annotations to
  downstream users.
- **[ts]** `DEFAULT_MODEL` constant in `src/defaults.ts` replaces
  hardcoded `"gpt-5-mini"` fallbacks across the codebase (debate
  engine, judges, classifiers, persona registry, CLI flags). Parity
  with Python's `llm_jury/_defaults.py`.
- **[ts]** `costUsd` field on `ClassificationResult`. `LLMClassifier`
  now forwards `payload.costUsd`; `Jury` fast-path and pre-flight-skip
  verdicts use `primary.costUsd ?? 0` instead of dropping the cost.
  Parity with Python's `ClassificationResult.cost_usd`.
- Test coverage for previously-untested branches: empty-personas
  debate (T1), TS calibration edge cases (T8), single-persona /
  no-response-rounds consensus (T9), LLM client timeout (T7), Python
  HuggingFace + Sklearn adapters (T4), cascade failures mid-debate
  (T3), summariser failure (T5), malformed persona JSON (T6).
- Governance files: `CONTRIBUTING.md`, `CHANGELOG.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue templates.
- Troubleshooting sections in the root README and both package
  READMEs covering auth, parse fallback, 429 exhaustion, both
  cost-guard markers, cost-tracking gaps, empty personas, and the
  TS-specific 60s timeout / silent-logger gotchas.
- **Lint gates (C1b)**. `ruff` + `black` (Python, configured in
  `pyproject.toml`) and `eslint` (TypeScript, flat config in
  `eslint.config.js`) now run as a dedicated `lint` job in CI.
  `pip install -e ".[dev]"` brings the Python tooling in.
- **Response cache (F3)**: opt-in `CachingLLMClient` in both SDKs.
  LRU wrapper around any `LLMClient`, keyed on
  `(model, system_prompt, prompt, temperature, response_format)`.
  Configurable `max_size` (default 1000) and optional `ttl_seconds`.
  Successful responses only — exceptions propagate uncached.
  Exposes `hits`, `misses`, `size`, and `clear()`. No behaviour
  change unless wrapped explicitly.
- **Cost pre-estimate gate (F4)**: new optional `on_cost_estimate` /
  `onCostEstimate` callback on `Jury`. Fires with
  `(estimated_max_debate_cost_usd, text)` immediately before a
  debate would run — *before* the existing `max_debate_cost_usd`
  hard guard. Return `False` to skip the debate (verdict gets
  `judge_strategy="cost_guard_user_override"`); return `True` /
  `None` to proceed. Lets you layer per-tenant budgets, time-of-day
  gates, or any other policy on top of the fixed cap. The
  `Jury.estimated_max_debate_cost_usd` property (already public
  since R2) returns the heuristic upper-bound estimate
  `N_personas × max_rounds × estimated_cost_per_persona_usd`.
- **Confidence-based early stop (F7, closes R5)**: opt-in
  `DebateConfig.early_stop_min_confidence` /
  `earlyStopMinConfidence`. When set, the DELIBERATION loop also
  halts after any round whose **minimum** persona confidence is
  `>=` this threshold, even if personas disagree on label. The
  original unanimous-label consensus check still triggers early
  exit regardless. Defaults to disabled (no behaviour change).

### Changed
- **[py]** Two `zip(...)` call sites in
  `calibration/optimizer.py` and `debate/engine.py` now pass
  `strict=True` to make their same-length invariants explicit.
  Same runtime behaviour when invariants hold; raises `ValueError`
  immediately if they ever don't (instead of silently truncating).
- **[py]** Five `setattr(obj, "literal", value)` calls in test
  helpers simplified to direct attribute assignment.
- **[py]** Unused `best = asyncio.run(...)` assignment dropped in
  CLI `calibrate` (calibrator mutates `jury.threshold` in place).
- **[py]** 32 files reformatted by `black` (whitespace / line wrap
  only — no semantic changes).

### Changed
- **[py][ts]** `FakeLLMClient` test helper now prefers a
  `system_prompt` match over a user-prompt match when routing
  per-persona responses. The user prompt in deliberation rounds
  legitimately mentions other personas ("Persona A said …"),
  which previously caused the first-iterated key to win for
  every persona. Pure test-infra change — no production-code
  impact, no production behaviour change.

### Fixed
- **Failed persona calls no longer count as votes** (both SDKs).
  Previously a persona whose LLM call failed (missing API key,
  provider outage, rate-limit exhaustion) or whose output couldn't
  be parsed was recorded as a real `labels[0]` vote at confidence
  0.0. With every persona failing, `MajorityVoteJudge` fabricated a
  unanimous `labels[0]` verdict at confidence **1.0** (silently
  flipping e.g. an "unsafe" primary classification to "safe"), and
  `BayesianJudge` elected the *opposite* label at ~1.0. Failed
  responses are now marked `failed=True` / `failed: true`, stay in
  the transcript for audit, and are excluded from all four judges,
  the consensus check, and every prompt builder. When the entire
  final round failed, judges fall back to the primary classifier
  result (`LLMJudge` additionally skips its own doomed LLM call,
  marker `llm_judge_fallback_personas_failed`).
- **Deliberation debates stop paying after a dead opening round**
  (both SDKs). If every persona call in the opening round failed,
  the engine no longer runs further deliberation rounds and the
  summariser against pure failure placeholders (previously 7 doomed
  LLM calls for a 3-persona debate; now 3). An all-failed later
  round likewise halts remaining rounds.
- **One bad row no longer destroys a whole CLI batch** (both SDKs).
  `llm-jury classify` now records per-row failures as
  `{"text", "error"}` rows in the output JSONL, keeps the verdicts
  that succeeded, warns on stderr, and exits non-zero only when
  every row failed.
- **Summariser failure no longer crashes the verdict** (both SDKs).
  If the summarisation LLM call raises, the engine logs a warning and
  returns the transcript with `summary=None`/`undefined`. Persona
  rounds are the load-bearing output. Matches the per-persona
  fallback pattern already used in `_run_round`.
- Latent test-fixture bug: `_FlakyLLMClient` in Python tests didn't
  accept `response_format`, so deliberation-mode failure tests were
  passing for the wrong reason (every persona was failing with
  `TypeError`, not just the targeted one). Fixture now matches the
  real client signature; tests assert non-failed personas succeed.

## [0.1.1] — 2026-04 / 2026-05 (initial published versions)

The initial released version of both SDKs. Highlights of what landed
before this CHANGELOG was started, ordered from oldest to newest:

### Core feature work
- Python SDK: confidence-driven escalation middleware with persona
  debate (independent / sequential / deliberation / adversarial
  modes), pluggable judge strategies (majority vote, weighted vote,
  Bayesian, LLM judge), threshold calibrator.
- TypeScript SDK: full parity port with native type safety.
- Examples for both SDKs (content moderation, custom personas, legal
  compliance, threshold calibration). TS examples type-checked in CI.
- Structured-output enforcement (F2): both SDKs build a JSON Schema
  via `build_persona_response_schema(labels)` and pass it as
  `response_format` to `LLMClient.complete()`.

### Reliability hardening
- TypeScript `classifyBatch` race condition fixed — semaphore +
  per-task `Promise.all` so workers no longer duplicate / skip texts.
- `asyncio.gather` / `Promise.all` over personas wrapped with
  `return_exceptions=True` + per-task fallback. A single juror
  failing no longer crashes the verdict.
- Empty-labels guard in Python `LLMClassifier`.
- Verdict provenance (`library_version`, `created_at`) and TS
  `Verdict` upgraded from `type` to class with `toDict()` / `toJSON()`.
- Pre-flight cost guard: estimated debate cost is checked *before*
  spending, refusing the debate when it would obviously blow the cap.
- 429 / 5xx retry hardening via structured error inspection
  (`isRetryableError`) on both SDKs.
- Judge fields (`was_escalated`, `primary_result`, `debate_transcript`,
  `total_duration_ms`) backfill only when at default — no more silent
  overwrite of custom-judge values.

### Tooling / CI
- Structured logging hooks in TypeScript (parity with Python's
  `logger`).
- `npm run check` (tsc) is a CI gate.
- Examples gate: `npm run check:examples` type-checks the runnable
  TS examples against the built package surface.
- Lock files committed for reproducibility.

[Unreleased]: https://github.com/mokhld/llm-jury/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mokhld/llm-jury/releases/tag/v0.1.1
