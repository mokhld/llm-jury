# llm-jury — Repository Audit

> **Date:** 2026-05-21
> **Scope:** Python SDK (`llm-jury-classifier`), TypeScript SDK (`@llm-jury/core`), examples, CI/CD, docs, cross-cutting concerns (security, reliability, observability, cost).
> **Method:** Four parallel investigation agents (Python audit, TypeScript audit, docs/CI audit, cross-cutting audit) followed by synthesis.

---

## Executive Summary

`llm-jury` is a well-architected dual-SDK (Python + TS) library implementing the LLM-jury classification pattern with personas, debate modes, and pluggable judge strategies. The Python package is the source of truth and is structurally clean. The TypeScript port is ~85% feature-complete with one **critical correctness bug** (`classifyBatch` race condition), missing serialization, and no logging.

**Top issues that warrant immediate attention:**

1. **`classifyBatch` race condition in TypeScript** — non-atomic cursor, workers can duplicate or skip texts. `src/jury/core.ts:128-146`.
2. **`asyncio.gather` / `Promise.all` without exception handling in debate** — one juror failing crashes the entire verdict. `python/debate/engine.py:186-192`, `typescript/debate/engine.ts:213-217`.
3. **Cost budget checked *after* spending** — actual cost can exceed `max_debate_cost_usd` by N personas × M rounds. `python/jury/core.py:99-114`.
4. **No logging at all in TypeScript** — production debugging is blind. Python has logger usage; TS is silent.
5. **Verdict has no `model`/`seed`/`version` — verdicts are not replayable.** Risk for compliance/audit use cases.
6. **TypeScript `Verdict` is a plain type, not a class — no `to_json()`/`to_dict()` parity with Python.**

The library is **not yet production-hardened** for high-stakes use (compliance, moderation at scale, regulated industries) but the bones are solid.

---

## 1. Bugs (Correctness)

### Critical

| # | Location | Issue |
|---|----------|-------|
| B1 | `packages/typescript/src/jury/core.ts:128-146` | **Race condition** in `classifyBatch`. Non-atomic `cursor` read-then-increment lets parallel workers grab the same index → duplicated work / overwritten results. Python uses `asyncio.Semaphore` which is safe; TS needs a real queue or semaphore. |
| B2 | `packages/python/src/llm_jury/debate/engine.py:186-192`, `packages/typescript/src/debate/engine.ts:213-217` | `asyncio.gather()` / `Promise.all()` over personas with **no `return_exceptions=True`**. A single juror timeout / 429 / network blip kills the whole verdict. |
| B3 | `packages/python/src/llm_jury/classifiers/llm_classifier.py:41` | Fallback label uses `self.labels[0]` with **no guard for empty `labels`** → `IndexError` on misconfigured classifier. |

### High

| # | Location | Issue |
|---|----------|-------|
| ~~B4~~ | `packages/python/src/llm_jury/jury/core.py` | ~~After `judge.judge()` returns, fields `was_escalated`, `primary_result`, `debate_transcript`, `total_duration_ms` are overwritten **unconditionally**.~~ **Fixed**: Jury now backfills only when the field is at its default/unset value (None for refs, 0 for `total_duration_ms`). `was_escalated` remains Jury-authoritative (Jury knows we escalated). Default judges now emit `total_duration_ms=0` so Jury fills the true full-classify duration. |
| ~~B5~~ | `packages/typescript/src/debate/engine.ts:387` | ~~Hardcoded fallback model `"gpt-5-mini"` when `personas` is empty — diverges from Python's `DEFAULT_MODEL` constant.~~ **Fixed** (PR #13): new `packages/typescript/src/defaults.ts` exports `DEFAULT_MODEL`; all 7 hardcoded occurrences across the TS codebase (debate engine, judges, classifiers, persona registry, CLI flags) now import the constant. |
| ~~B6~~ | `packages/typescript/src/judges/llmJudge.ts` | ~~Cost accumulation coerces `null` `totalCostUsd` to `0`, losing actual cost when the field starts nullable.~~ **Fixed**: new `sumCosts(a, b)` helper returns `null` when both inputs are null/undefined (signaling cost tracking unavailable) and otherwise sums with unknowns treated as 0. |
| B7 | `packages/typescript/src/jury/core.ts:140` | If a `classifyBatch` worker throws, `Promise.all` rejects and the caller gets no partial results — no per-item error handling. |
| ~~B8~~ | `packages/python/src/llm_jury/llm/client.py` | ~~Tenacity retries on `ConnectionError`/`TimeoutError`/`OSError` only. Does not explicitly retry 429 / 503.~~ **Fixed**: custom `_is_retryable_error` predicate now also retries on litellm typed errors (matched by class name to avoid hard import) and any exception with `status_code` / `http_status` / `status` in {429, 500-599}. |
| ~~B9~~ | `packages/typescript/src/llm/client.ts` | ~~Retryable-error detection uses a regex `5\d{2}\|429` on the error message — fragile; many SDK errors don't surface status this way.~~ **Fixed**: new exported `isRetryableError` inspects `err.status` / `err.statusCode` / `err.code` / `err.response?.status`; the regex fallback is kept for back-compat. The HTTP error thrown on `!response.ok` now carries `.status`. |

### Medium

| # | Location | Issue |
|---|----------|-------|
| B10 | `packages/python/src/llm_jury/debate/engine.py:286` | `_summarise()` accesses `self.personas[0].model` with no empty-list guard. Defended upstream but not locally. |
| B11 | `packages/python/src/llm_jury/llm/client.py:65` | `int(usage.total_tokens or 0)` silently turns `None` into `0` — masks cost-tracking bugs. |
| B12 | `packages/typescript/src/classifiers/huggingFaceAdapter.ts:38` | Runtime shape detection via `Array.isArray(raw[0])` instead of discriminated union — fragile. |
| B13 | `packages/typescript/src/debate/engine.ts:119-120, 251-252` | `Number(response.tokensUsed ?? 0)` — string / NaN coerce silently. |
| B14 | `packages/python/src/llm_jury/utils.py:19-21` | `safe_json_parse()` discards arrays/scalars (only returns dicts) and swallows the parse error — debugging is hard. **Mitigated for persona responses by F2** (schema-constrained); still applies to `LLMClassifier` + other JSON consumers. |
| B15 | `packages/python/src/llm_jury/judges/llm_judge.py:70-72` | Assumes LLM response has `"label"`; if missing, silently falls back without surfacing the malformed response. (LLMJudge uses a separate code path; F2 only covers persona responses.) |
| B16 | `packages/python/src/llm_jury/debate/engine.py:159-162` | Summarization in DELIBERATION runs even if consensus reached on round 1 — wasted LLM call. |

### Low

| # | Location | Issue |
|---|----------|-------|
| B17 | `packages/python/src/llm_jury/jury/core.py:141` | `bool(self.escalation_override(result))` — no validation that the callable returns a bool. |
| B18 | `packages/python/src/llm_jury/judges/bayesian.py:21` | Uniform prior via `1.0/max(1,len(labels))` — correct but undocumented; should assert at boundary. |
| B19 | `packages/typescript/src/judges/weightedVote.ts:28-34` | Tie-breaker depends on Map iteration order — non-deterministic across engines. |
| B20 | `packages/typescript/src/debate/engine.ts:422` | `raw.slice(0, 200)` truncates parse-error context — keep full payload for debugging. |
| B21 | `packages/python/src/llm_jury/calibration/optimizer.py:69` | Validator at L36-37 checks length mismatch but not empty input. |

---

## 2. Security & Safety

| # | Issue | Mitigation |
|---|-------|-----------|
| S1 | **Prompt injection** — user text is interpolated raw into persona prompts (`python/debate/engine.py:81-85`, `typescript/debate/engine.ts:281`). Adversarial input can override system instructions. | Add length cap, optional escaping helper, document risk in README, recommend wrapping untrusted text in delimiters and using structured-output APIs. |
| S2 | **API keys may leak via exceptions / logs** — `litellm` errors can include request payloads. | Add a redaction wrapper around log/exception paths; document recommended logging hygiene. |
| S3 | **Silent JSON parse failures hide adversarial / malformed responses** (B14, B15). **Partially mitigated** by F2 (persona responses now constrained server-side via JSON Schema `response_format`), but `safe_json_parse` is still the fallback when providers don't enforce. `on_parse_error` callback / strict-mode flag remain open. | Add `on_parse_error` callback or strict-mode that raises instead of falling back. |
| S4 | **No input length cap** before sending to LLM — denial-of-wallet via large inputs. | Enforce `max_input_chars` with a clear error. |
| S5 | **No SECURITY.md / vulnerability disclosure policy.** | Add SECURITY.md with contact + supported-versions table. |

---

## 3. Reliability & Cost

| # | Issue | Where |
|---|-------|-------|
| R1 | Partial-failure crashes (B2) — single juror failure kills verdict. | `debate/engine.py:186-192`, `debate/engine.ts:213-217` |
| ~~R2~~ | ~~Cost budget checked **after** debate completes — actual cost can exceed cap by an entire round.~~ **Fixed**: pre-flight estimate (`estimated_cost_per_persona_usd` × N × max_rounds) refuses the debate when it would obviously blow the cap (`judge_strategy="cost_guard_pre_flight"`). TS `runRound`/`runDeliberationRound` also halt between concurrency-batches once the cap is exceeded. Known limitation: in-flight LiteLLM calls aren't reliably cancellable, so the cap can still be overshot by up to one concurrency-batch worth of spend. | `jury/core.py:106-128`, `jury/core.ts:113-126`, `debate/engine.ts:225-258` |
| R3 | No hard `max_tokens` on LLM calls — long transcripts can blow context and cost. | `llm/client.py`, `llm/client.ts` |
| R4 | TypeScript timeout hardcoded to 60s. | `llm/client.ts:82` |
| R5 | Consensus check uses label equality only — doesn't consider confidence; entropy-based early stop would cut cost. | `debate/engine.py:434` |
| R6 | No response cache. Identical `(model, prompt)` → re-queried every time. | global |
| R7 | 429s and 5xx now trigger retry (B8/B9), but the `Retry-After` server hint is still ignored — backoff is purely exponential. | `llm/client.py`, `llm/client.ts` |
| R8 | `classify_batch` builds all coroutines up-front — memory spike at 100k+ items. | `jury/core.py:130-137` |
| R9 | `Classifier.classify_batch()` base impl is sequential `await` in a list-comp — defeats batching unless every subclass overrides. | `classifiers/base.py:23-24` |

---

## 4. Observability & Reproducibility

| # | Gap |
|---|-----|
| O1 | **TypeScript has zero logging.** Python has `logger` usage in client/debate. Add a `logger` / `onDebug` hook to the TS `Jury` options. |
| O2 | `Verdict` does not capture `model`, `temperature`, `seed`, library version. **Verdicts are not replayable** — blocker for any compliance use case. Add a `provenance` field. |
| O3 | No tracing hooks (OpenTelemetry). Add an optional `onSpan` callback. |
| O4 | `JuryStats` only tracks total / fast-path / escalated. Add: per-persona error rate, mean rounds-to-consensus, mean cost/verdict, parse-error rate. |
| O5 | TypeScript `Verdict` is a plain `type` — no `toJSON()` / `toDict()` parity with Python's dataclass. Downstream serialization code from Python users will not transfer. |
| O6 | No way to export an audit log (JSONL / parquet) of verdicts + transcripts. |

---

## 5. API & Architecture Improvements

| # | Improvement |
|---|------------|
| A1 | **TypeScript `Verdict` should be a class** with `toJSON()` and `toDict()` methods (parity with Python). `packages/typescript/src/judges/base.ts:4-14` |
| ~~A2~~ | ~~Extract `fallbackVerdict()` helper in TS — current duplication across `majorityVote.ts:8-19`, `bayesian.ts:15-26`, `weightedVote.ts:8-19`.~~ **Fixed**: new `fallbackVerdict(transcript, judgeStrategy)` exported from `judges/base.ts`; the three judges now delegate to it. Python parallel is `_fallback_verdict()` in `judges/base.py:39-52`. |
| ~~A3~~ | ~~Add `cost_usd` to `ClassificationResult` base in TS (Python has it). `packages/typescript/src/classifiers/base.ts`~~ **Fixed** (PR #13): added optional `costUsd?: number \| null` to TS `ClassificationResult`; `LLMClassifier` forwards `payload.costUsd`; `Jury` fast-path and pre-flight-skip verdicts use `primary.costUsd ?? 0` instead of hardcoded 0. |
| A4 | `Persona` in TS should have default `model` / `temperature` like Python's dataclass. |
| A5 | `Jury` options object is large — consider a builder pattern for both languages, or at least a `JuryConfig.from_preset("content_moderation")` factory. |
| A6 | Consider a strict-mode flag (`strict_parsing=True`) that surfaces malformed-LLM-response errors instead of silently falling back. |
| A7 | Decouple `Persona.model` from a global default — require explicit or factory injection. `personas/base.py` |
| A8 | Tighten types in TS: `as unknown` cast chain in `llmJudge.ts:50` and `debate/engine.ts:406` should narrow with type guards. |
| A9 | `SklearnLikeModel.predictProba(features: unknown): number[][]` — validate shape or use a stricter type. `classifiers/sklearnAdapter.ts:3-5` |

---

## 6. Missing Features (Production users will want these)

| # | Feature |
|---|---------|
| F1 | **Streaming verdicts** — yield persona responses as they arrive (async generator / `AsyncIterable`). |
| ~~F2~~ | ~~**Structured-output enforcement** via JSON Schema — replaces fragile `safe_json_parse`.~~ **Fixed (persona path)**: both SDKs now build `build_persona_response_schema(labels)` (`personas/schema.py` / `personas/schema.ts`) and pass it as `response_format` to `LLMClient.complete()`. LiteLLMClient forwards it on Python/TS. `safe_json_parse` remains as defensive fallback. `LLMClassifier`'s own JSON parse path is intentionally out of scope here — separate follow-up. |
| F3 | **Response cache** (LRU keyed by `(model, system, prompt, temperature, seed)`) with TTL. |
| F4 | **Cost pre-estimate** — before running a debate, estimate `N_personas × M_rounds × avg_tokens × $/tok` so users can decide. |
| F5 | **Verdict replay** — given a captured provenance block, replay deterministically (requires O2). |
| F6 | **Async callbacks / webhooks** — emit a verdict to Kafka / HTTP endpoint when ready. |
| F7 | **Entropy-based early-stop** in debate — if confidence > 0.95 across personas, exit before `max_rounds`. |
| F8 | **Persona ensembles vs. routing** — allow a "router" mode that picks 1 of N personas based on input topic instead of running all. |
| F9 | **Tracing / OpenTelemetry hooks.** |
| F10 | **Audit-log exporter** (JSONL / parquet) — see O6. |
| F11 | **Prompt-injection detector** as an optional pre-classifier guard. |

---

## 7. Test Coverage Gaps

| # | Gap |
|---|-----|
| ~~T1~~ | ~~Empty `personas` list — `debate()` defends at L91 but never exercised by a test.~~ **Fixed** (PR #10): `test_empty_personas_returns_zero_round_transcript` in `tests/test_debate/test_modes.py`. |
| ~~T2~~ | ~~Empty `labels` list — would trigger B3 if it happened.~~ **Already covered** by `test_llm_classifier_rejects_empty_labels` (+ whitespace + None variants) in `tests/test_classifiers/test_llm_classifier.py`. |
| ~~T3~~ | ~~Cascade failures — what if LLM client raises mid-debate? Only mock-based integration test.~~ **Fixed** (PR #12): `test_failure_paths.py::CascadeFailureTests` — persona failing on round 2 + all-personas-fail-on-round-2. Also fixed a latent fixture bug where `_FlakyLLMClient` didn't accept `response_format`, so the original "one persona failure" tests were passing because *every* persona was failing. |
| ~~T4~~ | ~~HuggingFace and Sklearn adapters in Python — no unit tests.~~ **Fixed** (PR #11): 4 tests each in `tests/test_classifiers/test_huggingface_adapter.py` and `test_sklearn_adapter.py`. Inject fake `transformers` / `numpy` into `sys.modules` so tests run without the optional extras installed. |
| ~~T5~~ | ~~Summarization failure — if the summarization LLM call fails, no test asserts behavior.~~ **Fixed** (PR #12): the path was undefended in both SDKs; PR added try/except around `_summarise()` (Python `engine.py:165-172`) and try/catch around `summarise()` (TS `engine.ts:186-200`), plus test in `test_failure_paths.py::SummariserFailureTests` and `modes.test.ts`. Verdict now returns `summary=None`/`undefined` when summarisation raises. |
| ~~T6~~ | ~~Malformed persona JSON — `_parse_persona_response` handles it but isn't tested with missing `label` / out-of-bounds `confidence`.~~ **Fixed** (PR #12): `test_failure_paths.py::MalformedPersonaJSONTests` covers missing label, OOB confidence (clamped), non-numeric confidence (raises `ValueError` — pinned), non-string label (coerced), and array payload (fallback). |
| T7 | ~~TypeScript: no retry-exhaustion test, no transient-error-detection test, no timeout test.~~ **Mostly fixed**: retry/transient covered by PR #5 (`isRetryableError` properties + 503-then-success + 400-no-retry). Timeout added in PR #10 (`litellm client aborts a request that exceeds timeoutMs and retries`). Multi-attempt exhaustion still open but lower priority. |
| ~~T8~~ | ~~TypeScript: no calibration edge cases (empty input, single threshold, NaN).~~ **Fixed** (PR #10): three tests in `tests/calibration/optimizer.test.ts`. |
| ~~T9~~ | ~~TypeScript: no consensus tests for single-persona debate or no-response rounds.~~ **Fixed** (PR #10): single-persona deliberation + `consensusReached([])` tests in `tests/debate/modes.test.ts`. |
| ~~T10~~ | ~~Race-condition test for `classifyBatch` (B1) — would catch it.~~ **Already covered** by "classifyBatch preserves order and runs each input exactly once under high concurrency" in `tests/jury/core.test.ts` (50 inputs × concurrency=8, exactly-once + order). |

---

## 8. Docs, Examples, CI/CD

### Docs

| # | Issue |
|---|-------|
| D1 | Root `README.md:8-9` has **static "55 passed / 38 passed" badges** — will rot. Use dynamic badges from CI or remove. |
| D2 | No `CONTRIBUTING.md`. |
| D3 | No `CHANGELOG.md`. Release workflow uses `--generate-notes` but nothing is committed. |
| D4 | No `CODE_OF_CONDUCT.md`. |
| D5 | No `SECURITY.md` (S5). |
| D6 | No GitHub issue templates (`.github/ISSUE_TEMPLATE/`). |
| D7 | No troubleshooting section in any README. |

### Examples

| # | Issue |
|---|-------|
| ~~E1~~ | ~~**Zero TypeScript examples.** Python has 4.~~ **Fixed**: `examples/typescript/{content_moderation,custom_personas,legal_compliance,threshold_calibration}.ts` mirror the Python examples; type-checked in CI via new `npm run check:examples` (uses a separate tsconfig that points the `@llm-jury/core` alias at the built `dist/index.d.ts` so examples see the user-facing surface). |

### CI/CD

| # | Issue |
|---|-------|
| ~~C1~~ | ~~`ci.yml` runs tests only — **no lint (ruff/black/eslint), no type-check step** for TS.~~ **Partially fixed**: TS `npm run check` (tsc) is now a CI gate in `.github/workflows/ci.yml`. Lint (ruff/black/eslint) is still open — tracked as **C1b** in §9. |
| C2 | No dependabot config — supply-chain risk. |
| C3 | No prerelease / RC channel in `release.yml`. |
| ~~C4~~ | ~~No `py.typed` marker → Python type hints aren't exposed to downstream users.~~ **Fixed** (PR #13): added `packages/python/src/llm_jury/py.typed` and declared in `pyproject.toml [tool.setuptools.package-data]`; verified the marker lands in the built wheel. |
| C5 | `package.json` (TS) — consider `"sideEffects": false` to help bundlers tree-shake. |
| C6 | TS package.json `exports` is ESM-only — fine, but document it; some downstream CJS users will be surprised. |

### Repository hygiene

| # | Issue |
|---|-------|
| H1 | `package-lock.json` and `packages/python/uv.lock` are **untracked**. Decide: commit (for reproducibility) or add to `.gitignore`. Today they're in a limbo state. |
| H2 | `.gitignore` is otherwise reasonable. |

---

## 9. Prioritized Roadmap

### P0 — fix before next release ✅ all landed

- ~~B1~~: TS `classifyBatch` race → semaphore + per-task `Promise.all` (commit 04c70e5).
- ~~B2~~: `gather` / `Promise.all` swallow per-juror errors → `return_exceptions=True` + per-task fallback (commit 04c70e5).
- ~~B3~~: empty-labels crash in Python `LLMClassifier` → validation (commit 04c70e5).
- ~~O2~~: `Verdict` provenance — `library_version`, `created_at`, TS class with `toDict()`/`toJSON()` (commit 04c70e5).
- ~~D1~~: dynamic CI badge (commit 04c70e5).
- ~~H1~~: lock files committed (commit 04c70e5).

### P1 — next sprint

- ~~R2~~: cost guard before-the-fact, not after — pre-flight estimate + per-batch guard (PR #4).
- ~~O1~~: structured logging hooks in TS (parity with Python) (commit 85e98d2).
- ~~A1~~: TS `Verdict` class (landed in P0).
- ~~B4, B6~~: judge-field overwrite + TS cost coercion (PR #6).
- ~~B8, B9~~: 429/5xx retry hardening (PR #5).
- ~~A2~~ — extracted `fallbackVerdict()` helper in TS (judges/base.ts); majorityVote / bayesian / weightedVote now delegate.
- ~~C1~~ (tsc) — `npm run check` is now a CI gate.
- **Open:** C1b — lint gates (ruff/black for Python, eslint for TS). Net-new tooling — none of these are installed in the project today. Separate sprint because it surfaces unknown fix volume.
- ~~E1~~ — TypeScript versions of the 4 Python examples in `examples/typescript/`; type-checked in CI via `npm run check:examples`.
- ~~F2~~ — structured-output JSON Schema for persona responses (`personas/schema.{py,ts}` + `response_format` threaded through `complete()` in both SDKs). `LLMClassifier`'s own JSON parsing not yet covered (separate follow-up if desired).

### P2 — quality polish

- F3 (cache), F4 (cost pre-estimate), F7 (entropy early stop).
- ~~T1–T10: fill test gaps.~~ **All closed** (PR #10, #11, #12). T7 retry-exhaustion subset still open but low priority.
- D2–D6: governance files.
- A5: builder / preset factories.
- ~~C4: `py.typed`.~~ Closed in PR #13.
- C1b: lint gates (ruff/black for Python, eslint for TS) — net-new tooling, surfaces unknown fix volume.
- ~~A3: `cost_usd` on TS `ClassificationResult` base (Python has it).~~ Closed in PR #13.
- ~~B5: hardcoded `"gpt-5-mini"` fallback in TS diverges from Python's `DEFAULT_MODEL`.~~ Closed in PR #13.

### P3 — stretch / production-grade

- F1 streaming, F5 replay, F6 webhooks, F9 OTel, F10 audit export.
- F11 prompt-injection detector.
- F8 router mode.

---

## 10. Notes on the "Parity" Claim

The original TS-port commit message said "full parity port with native type safety." After the P0 and most of the P1 work, that claim is now substantially true. Remaining asymmetries:

- ~~`Verdict` is a `type` not a class~~ → **fixed (P0).**
- ~~No logging~~ → **fixed (O1).**
- ~~`classifyBatch` has a correctness bug Python doesn't~~ → **fixed (P0).**
- ~~`ClassificationResult.cost_usd` is still missing in TS base (audit A3)~~ → **fixed (PR #13).**
- ~~Judge fallback paths duplicated~~ → **fixed (A2).**
- ~~Hardcoded `"gpt-5-mini"` fallback diverges from Python's `DEFAULT_MODEL`~~ → **fixed (PR #13).**

All known parity asymmetries between the two SDKs are now closed.
