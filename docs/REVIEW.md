# llm-jury review

Review date: 2026-09-24. Reviewed commit: `02e63be` (v0.2.0 plus the npm OIDC CI change).
Fixes for the items marked `in progress` were started the same day.

This file is the living record of known problems and feature priorities. Update item
statuses in place and add new findings under a dated heading; do not delete history.
Feature briefs that an agent can implement directly live in [FEATURES.md](FEATURES.md).

Status values: `open`, `in progress`, `fixed` (with date), `wontfix` (with reason).

## Earlier audit (AUDIT.md)

README.md, CONTRIBUTING.md, SECURITY.md and CHANGELOG.md cite an `AUDIT.md` with item IDs
(B4, B6, B8, B9, R2, R5, R7, S1, S3, S4, F2, F3, F4, F7, F11, C1, C1b, C2, C4, T1-T9,
D2-D7, E1, A2, A3). That file was never committed (not in git history). What the IDs
meant, where it can be inferred from CHANGELOG and commit messages:

| ID | Meaning | State at review |
|---|---|---|
| B4, B6 | Jury stopped clobbering judge fields; null cost preserved | fixed in 0.1.1 |
| B8, B9 | Retry on 429/5xx via structured error inspection | fixed in 0.1.1 |
| R2 | Pre-flight cost guard | done |
| R5 | Early stop in deliberation | "closed" by F7, but see BUG-06 |
| R7 | Honour Retry-After | open, folded into BUG-08 |
| S1 / F11 | Prompt injection; built-in detector | open, see BUG-09 |
| S3 | LLMClassifier has no structured output | open, folded into BUG-01 |
| S4 | Denial of wallet via large inputs | open, no input cap exists |
| C2 | Dependabot | open |
| F2, F3, F4, F7 | Structured persona output, response cache, cost-estimate gate, confidence early stop | shipped in 0.2.0 |

New IDs in this file use the prefixes `BUG-`, `CLI-`, `AD-`, `CAL-`, `REL-`, `DOC-`,
`PKG-`, `CI-` (problems) and `FEAT-` (features) so they cannot collide with the old ones.

## Summary of the package

**What it does.** llm-jury is a middleware SDK shipped twice: Python
(`llm-jury-classifier` on PyPI, import `llm_jury`) and TypeScript (`@llm-jury/core` on
npm), kept as parity ports. It wraps any classifier that returns `(label, confidence)`.
Confident results return directly (fast path). Results below `confidence_threshold`
escalate to a panel of LLM personas that debate in one of four modes (independent,
sequential, deliberation = the 4-stage CEJ pipeline from arXiv:2512.23732, adversarial).
A judge (LLM, majority, weighted, Bayesian) turns the debate into a `Verdict` carrying
the full transcript. Extras: cost guards, a cost-estimate callback, an LRU response
cache, a threshold calibrator, and a JSONL batch CLI.

**Value proposition.** Stated in the README: "When your classifier is uncertain, let a
configurable jury of LLM personas debate and return an auditable verdict." Inferred from
code and docs, users are promised: (1) a better, valid label on ambiguous inputs,
(2) spend only on uncertain items, bounded and reported honestly, (3) an audit trail they
can trust, (4) production reliability (degrade instead of crash), (5) identical behaviour
in Python and TypeScript.

**Users.** ML and platform engineers running production classification pipelines
(content moderation / trust and safety, legal and financial compliance, triage) who want
low-confidence cases resolved automatically with a rationale instead of a human queue.
They rely on the label being one of their labels, the cost cap holding, and the
transcript being an honest record.

## Part A: what doesn't work (ranked)

Evidence key: VERIFIED means reproduced with a script or traced line by line;
INFERRED means read from code or docs but not executed. Reproduction scripts used fake
LLM clients (no network).

### A1. BUG-01 Verdict labels are never validated against the configured labels
- Severity: critical. Status: in progress.
- Where: `judges/llm_judge.py:81` / `judges/llmJudge.ts:100` (default judge, no
  `response_format`, accepts any string); persona parse `debate/engine.py:579` /
  `engine.ts:560`; `classifiers/llm_classifier.py:59` / `llmClassifier.ts:62`.
  Majority and weighted judges count whatever labels personas return; the persona JSON
  schema enum only helps when the provider enforces strict `json_schema`.
- Evidence (VERIFIED, both SDKs): labels `[safe, unsafe]`; LLM judge answering
  `"Unsafe - borderline"` produced `verdict.label == "Unsafe - borderline"`; personas
  answering `"UNSAFE"` gave a majority verdict `"UNSAFE"` at confidence 1.0 with
  `debate_degraded=False`; an LLMClassifier answering `"SAFE"` at 0.99 took the fast
  path with label `"SAFE"`.
- Why it matters: pipelines route on the label. An unknown label silently falls through
  routing, after the debate was paid for. This is the default path.
- Fix: canonicalise with a shared `match_label`; out-of-set persona labels count as
  failed responses; send label-enum schemas from `LLMJudge` and `LLMClassifier`; judge
  falls back to a vote over valid responses (`llm_judge_fallback_invalid_label`).

### A2. BUG-02 Non-numeric or NaN confidence skips escalation
- Severity: high. Status: in progress.
- Where: TS `llmClassifier.ts:63`, `engine.ts:561`, `llmJudge.ts:101` use bare
  `Number(...)`; `jury/core.ts:253` escalates on `confidence < threshold`, which is false
  for NaN. Python `utils.clamp_confidence` turns NaN into 1.0 and raises on `"low"`.
- Evidence (VERIFIED): TS LLMClassifier returning `confidence: "low"` gave
  `confidence: NaN, wasEscalated: false`. TS personas returning `"high"` made
  WeightedVote return label `""` at confidence -1. Python `json.loads` accepts `NaN`,
  which clamps to 1.0 (fast path); a Python LLMClassifier returning `"low"` raised
  `ValueError` out of `classify`.
- Why it matters: routing uncertain items is the product's core job, and it fails open.
- Fix: shared `parse_confidence` (None for non-finite / non-numeric); classifier treats
  None as 0.0 (escalates); personas as failed; judge falls back; `_should_escalate`
  escalates non-finite confidences; validate `confidence_threshold` in [0, 1].

### A3. BUG-04 Cost reporting and the cost cap are unreliable
- Severity: high. Status: in progress.
- Where: unknown per-call cost is coerced to 0 (`engine.py:378, 424`, `engine.ts:147` and
  others). The TS `LiteLLMClient` never reports cost (`llm/client.ts:151-155`). The
  pre-flight estimate (`jury/core.py:72-79`, `core.ts:84-90`) counts persona calls only,
  not the summariser or judge, and uses `max_rounds` even in single-round modes.
  `CachingLLMClient` returns the stored cost on hits. Escalated verdicts drop the primary
  classifier's cost.
- Evidence (VERIFIED): with `cost_usd=None` and `max_debate_cost_usd=0.07`, all 8 calls
  ran and the verdict reported `total_cost_usd=0.0`. In TS with the default client shape
  and a $0.0001 cap, 8 calls ran and `totalCostUsd` was 0. A cache hit with no new spend
  reported the full cost again. A primary costing 0.5 plus 3 persona calls at 0.001
  reported 0.003. With a cap equal to the estimate, float noise
  (`0.060000000000000005 > 0.06`) tripped the guard after the debate finished and
  discarded it.
- Why it matters: "bounded spend" rests on a flat $0.01-per-call heuristic in TS by
  default and in Python for unpriced models, and dashboards show $0. SECURITY.md tells
  users to rely on `max_debate_cost_usd`.
- Fix: keep unknown cost as None; count `unpriced_calls` and charge them at the per-call
  estimate against the cap; estimate every call; zero-cost cache hits; add primary cost
  to escalated totals; compare with a small tolerance. Pricing for TS is FEAT-04.

### A4. BUG-03 A judge failure throws away the paid debate
- Severity: high. Status: in progress.
- Where: `jury/core.py` calls `self.judge.judge(...)` with no handling; `LLMJudge.judge`
  lets the LLM exception escape (`llm_judge.py:57`, `llmJudge.ts:79`, `core.ts:193`).
- Evidence (VERIFIED, both SDKs): a judge 400/503 after 7 paid debate calls rejected
  `classify`. With default `classify_batch` the whole batch rejects.
- Why it matters: the most expensive work is lost on a transient judge error, and
  persona failures degrade gracefully while judge failures crash.
- Fix: fall back to a majority vote over the final valid round
  (`llm_judge_fallback_error`), then to the primary result.

### A5. CAL-01 Calibration produces meaningless thresholds
- Severity: high. Status: in progress (CLI-01 fix; FEAT-01 rewrites calibration).
- Where and evidence (VERIFIED):
  - CLI `function` classifier falls back to the ground-truth `label` when
    `predicted_label` is missing (`cli/main.py:105`, `cli/main.ts:135`). `llm-jury
    calibrate` on rows with `text` + `label` reported accuracy 1.0 and escalation 0.0 at
    every threshold, recommending 0.5 (never escalate).
  - TS `calibration/optimizer.ts:41-53` re-classifies every text for every threshold
    (10x the primary spend with the 10 default thresholds) and counts escalated items as
    correct (accuracy 1.0 with an always-wrong classifier). Python caches and excludes
    them, so the SDKs disagree on the same data.
  - Neither SDK ever runs the jury during calibration; the CLI's `--personas`, `--judge`,
    `--debate-mode`, `--max-rounds`, model and cost flags do nothing for `calibrate`.
- Why it matters: the threshold is the main cost/quality knob, and it is tuned on
  leaked or invented numbers. Nobody can show that the jury beats the primary
  classifier on their data.
- Fix: require `predicted_label` / `predicted_confidence` (CLI-01); classify once and
  exclude escalations in TS; measure the jury (FEAT-01).

### A6. CLI-02 The npm-installed TypeScript CLI does nothing and exits 0
- Severity: high (critical for anyone running the TS CLI in a pipeline). Status: in progress.
- Where: `packages/typescript/src/cli/main.ts:316` runs `main()` only when
  `import.meta.url === "file://" + process.argv[1]`. npm installs bins as symlinks.
- Evidence (VERIFIED): running `dist/cli/main.js` through a symlink printed nothing;
  the infra review packed and installed the tarball and `npx llm-jury classify ...`
  exited 0 without writing output. Related: `--version` prints hard-coded `0.1.0`
  (CLI-03); non-numeric numeric flags hang or query zero personas (CLI-04); output rows
  omit `debate_degraded` (CLI-05).
- Why it matters: batch jobs report success with no output. Unit tests call `main()`
  directly, so CI cannot catch it.
- Fix: realpath comparison or a dedicated bin file; test through a symlink; CI smoke test
  of the packed tarball (CI-01).

### A7. BUG-09 Untrusted input is pasted into prompts without delimiters
- Severity: high for content-moderation users. Status: in progress.
- Where: `debate/engine.py:441` (`## Input to Classify\n\n{text}`), the deliberation and
  summariser prompts, `llm_judge.py:97`, `llm_classifier.py:38`, and the TS equivalents.
- Evidence (VERIFIED by reading; impact INFERRED): the input sits inside the same
  markdown structure as the instructions, so text containing `## Primary Classifier
  Result` or instructions is indistinguishable from the prompt. Escalated items are the
  borderline ones an attacker can craft. SECURITY.md declares injection out of scope and
  says "mitigation guidance is in the README"; no README contains any.
- Fix: wrap input in `<input>` tags with an "untrusted data" note and neutralise tag
  look-alikes; write the README guidance; consider an input length cap (old S4).

### Other confirmed problems

| ID | Severity | Problem | Evidence | Status |
|---|---|---|---|---|
| REL-01 | high | `release.yml:97-115` commits to main and force-pushes the tag before tests run, also on `dry_run`; re-running moves an existing tag. v0.2.0 was published to PyPI from `8809991` and to npm from `02e63be`. Release tests skip tsc and lint and CI status. | VERIFIED (workflow read, `gh run` history) | open |
| DOC-01 | high | Root README Overview says the Python package is `llm-jury`; that name on PyPI is an unrelated project that ships the same `llm_jury` import package. | VERIFIED | open |
| AD-01 | high | TS `HuggingFaceClassifier` calls the pipeline without top-k options and freezes `labels` from the first call's single top result, so debates run over one allowed label. No `labels` option (Python has one). | VERIFIED with injected pipeline; library default INFERRED | in progress |
| BUG-05 | medium | `on_verdict` / `onVerdict` fires only on judged verdicts, not fast path or cost-guard verdicts (most traffic). | VERIFIED both SDKs | in progress |
| BUG-06 | medium | Consensus and `early_stop_min_confidence` are checked only after rounds 2+, so at default `max_rounds=2` early stop never saves a call; a unanimous opening round still pays for round 2 and the summariser. | VERIFIED both SDKs | in progress |
| BUG-07 | medium | Fail-fast `classify_batch` keeps starting debates after it rejects (TS: 0 calls at rejection, 12 calls 50 ms later). | VERIFIED TS, INFERRED Python | in progress |
| BUG-08 | medium | Python `LiteLLMClient` has no timeout (litellm default `request_timeout=6000` s) and takes no constructor args, so README's `LiteLLMClient(api_key=...)` raises TypeError. TS retries 4xx errors whose body contains 5xx-looking numbers; neither SDK honours Retry-After; TS default client gets no logger; `openai/gpt-5-mini` style names still get a temperature. | VERIFIED (timeout value, TypeError, regex); prefix INFERRED | in progress |
| BUG-10 | medium | `Persona.known_bias` never reaches any prompt although the judge is told to weigh it; LLM judge's `key_agreements`, `key_disagreements`, `decisive_factor` are requested and discarded. | VERIFIED | in progress |
| AD-02 | medium | Python HF and sklearn adapters run blocking inference inside `async def`, stalling the event loop. | INFERRED from code | in progress |
| AD-03 | medium | Sklearn adapters map `predict_proba` columns to `labels` by position, ignoring `classes_` order. | INFERRED from code | in progress |
| DOC-02 | medium | README snippets that fail: `DebateMode.Independent` (real key `INDEPENDENT`; in plain JS it silently runs deliberation and makes the pre-flight estimate NaN); `LiteLLMClient(api_key=...)`; TS error strings in troubleshooting tables; "failed personas are dropped" (they stay with `failed=True`); cost docs claim `undefined`/`None` where code returns 0. | VERIFIED | open |
| DOC-03 | medium | Missing `AUDIT.md` cited in README, CONTRIBUTING, SECURITY; SECURITY.md supports only 0.1.x; CHANGELOG has two `### Changed` headings and stale compare links; CONTRIBUTING cites `examples/python/` (does not exist) and a `uv sync` flow that lacks pytest. | VERIFIED | open |
| PKG-01 | medium | No LICENSE in wheel, sdist or npm tarball; package READMEs link `../../LICENSE`; no `engines` field; `exports` has no `require`/`default` condition. | VERIFIED | open |
| PKG-02 | medium | `typer>=0.9` is too low: typer <=0.12.3 crashes the CLI at start; 0.13-0.15.4 with click 8.3 also crash. `litellm>=1.0` has no upper bound. | VERIFIED by the infra review | open |
| CI-01 | medium | CI tests editable installs and TS sources, never the built wheel or packed tarball; Node matrix is 22 only; lockfiles unused (`npm install`, not `npm ci`); unquoted `tests/**/*.test.ts` glob only matches one directory level. | VERIFIED | open |
| DOC-04 | medium | Latency and cost claims (README sample 2.8 s / $0.001; value-analysis 2-5 s / $0.001-0.005) disagree with the Feb 2026 live runs in `.test-artifacts/` (27-57 s, $0.007-0.015 per escalation). | VERIFIED against local artifacts | open |

## Part B: features to add or extend (ranked)

Agent-ready briefs with API sketches and acceptance criteria are in
[FEATURES.md](FEATURES.md). Summary:

| Rank | ID | Feature | Serves | Effort | Status |
|---|---|---|---|---|---|
| 1 | FEAT-01 | Jury evaluation harness, and calibration that uses measured jury outcomes | Promise 1 made measurable; the threshold reflects real jury value | medium-large | in progress |
| 2 | FEAT-02 | Human-review routing policy (`needs_review`, `review_reasons`, typed resolution) | The core user story: resolve what is safe, route the rest | small | open |
| 3 | FEAT-03 | Better routing signal: score distributions, top-2 margin routing, structured and logprob-based LLMClassifier confidence | Escalating the right items (paper uses margin routing) | medium | open |
| 4 | FEAT-04 | Budgets that bind: TS token pricing, auto per-call estimate, per-debate time budget | Cost and latency promises in TS and for unpriced models | medium | open |
| 5 | FEAT-05 | Observability hooks and dollar-based stats (per-call events, overturn rate, optional OpenTelemetry) | Audit and production operations | medium | open |
| 6 | FEAT-06 | Streaming, resumable batch (`classify_stream`, CLI append-as-you-go, `--resume`, `--personas-file`) | Batch users; a crash no longer loses paid work | medium | open |
| 7 | FEAT-07 | Reproducible verdict provenance (config snapshot, prompt hashes, template version) | Compliance users reproducing a decision | small | open |

Why this order: FEAT-01 comes first because every other claim (the jury improves
accuracy, the threshold is right, a persona set helps) is currently unmeasurable, and the
live artifacts already contradict the documented cost and latency. FEAT-02 is small and
turns the existing degraded/fallback signals into one routing decision users can act on.
FEAT-03 and FEAT-04 improve what escalates and what it costs, and both need FEAT-01 to
measure. FEAT-05 to FEAT-07 matter for production and compliance but assume the numbers
they report are right.

Considered and ranked lower: a sync `classify_sync` wrapper (ergonomics only), native TS
provider clients (an OpenAI-compatible endpoint or LiteLLM proxy already works), more
persona libraries (unmeasurable until FEAT-01), fast-path drift sampling (follows FEAT-01
and FEAT-05), adversarial-mode improvements (no rebuttal round, stance never names a
label; also unmeasurable until FEAT-01), a prompt-injection detector (old F11; delimiting
in BUG-09 comes first).

## Coverage

Covered in depth: both SDKs' source (jury, debate engine, judges, LLM clients, cache,
classifiers, calibration, CLI, personas), CI and release workflows, packaging (built the
wheel and packed the npm tarball), all READMEs and governance docs, examples (TS
type-check, Python calibration example run), and the local `.test-artifacts/` and
`.content/` notes. Both suites passed at review time: Python 139 passed / 1 skipped,
TypeScript 116 passed / 1 skipped.

Covered lightly or not at all:
- Real LLM behaviour (no live calls were made): whether providers honour the persona
  `json_schema`, litellm's `response_format` passthrough for non-OpenAI providers, real
  latency and cost on 0.2.0.
- Prompt quality and debate effectiveness (whether deliberation beats independent mode,
  persona set quality). Needs FEAT-01.
- `BayesianJudge` maths beyond reading (priors are multiplied once per persona; likelihood
  for non-chosen labels is `1 - confidence` for every other label in multi-class).
- Adversarial mode (`self.personas.index(persona)` assigns both sides the same stance if
  two personas compare equal).
- Windows, Node versions other than 22 and 24, Python 3.14 beyond one wheel test run.
- Memory growth of `CachingLLMClient` under large prompts (size-bounded by count, not bytes).
