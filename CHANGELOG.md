# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file covers both SDKs (Python `llm-jury-classifier`, TypeScript
`@llm-jury/core`). Where a change applies to only one SDK it is
marked **[py]** or **[ts]**.

## [Unreleased]

### Added
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

### Fixed
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
