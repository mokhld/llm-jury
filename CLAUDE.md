# llm-jury

## What it is

Middleware SDK, shipped twice as parity ports: Python `llm-jury-classifier` (PyPI, import
`llm_jury`) and TypeScript `@llm-jury/core` (npm). It wraps any classifier returning
`(label, confidence)`. Confident results return directly (fast path); results below
`confidence_threshold` escalate to a debate among LLM personas, and a judge turns the
debate into a `Verdict` with the full transcript. Based on the CEJ module of
arXiv:2512.23732.

Value proposition (README): "When your classifier is uncertain, let a configurable jury of
LLM personas debate and return an auditable verdict." Users are ML/platform engineers
running production classification (content moderation, compliance, triage). They rely on:
the verdict label being one of their labels, spend being bounded and reported honestly,
the transcript being an honest audit record, and verdicts degrading instead of crashing.

Known problems and their status: `docs/REVIEW.md`. Feature backlog with agent-ready
briefs: `docs/FEATURES.md` (point an agent at it to implement the next feature).

## Map

```
packages/python/src/llm_jury/        source of truth
  jury/core.py         Jury: routing, cost gates, callbacks, stats (entry point)
  debate/engine.py     DebateEngine, DebateConfig, DebateMode, DebateTranscript, prompts
  judges/base.py       Verdict dataclass, JudgeStrategy, fallback helpers
  judges/              llm_judge (default), majority_vote, weighted_vote, bayesian
  classifiers/         ClassificationResult/Classifier; function, llm, huggingface, sklearn
  personas/            Persona, PersonaResponse, registry (4 built-in sets), schema (JSON schema)
  llm/client.py        LLMClient protocol, LiteLLMClient (litellm + tenacity retry)
  llm/cache.py         CachingLLMClient (opt-in LRU)
  calibration/         ThresholdCalibrator
  cli/main.py          typer app: `llm-jury classify|calibrate` (JSONL in/out)
  utils.py, _defaults.py (DEFAULT_MODEL), _version.py
packages/typescript/src/             same layout, camelCase file names
  logger.ts (NOOP_LOGGER default), defaults.ts, _version.ts (LIBRARY_VERSION)
  llm/client.ts        fetch-based OpenAI-compatible client, 60 s timeout, own retry
packages/*/tests/                    mirror src; helpers.py / helpers.ts hold FakeLLMClient
examples/                            *.py and typescript/*.ts (TS ones type-checked in CI)
.github/workflows/                   ci.yml (lint, py 3.10-3.13, node 22), release.yml
```

Gitignored, local only: `.content/` (articles, `value-analysis.md` on paper fidelity),
`.test-artifacts/` (Feb 2026 live-API runs with real latency and cost). The original
product spec is `../llm-jury-spec.md`, outside the repo.

Request flow (`Jury.classify`): primary `classifier.classify` -> `_should_escalate`
(confidence < threshold, or `escalation_override`; empty `personas` disables escalation)
-> fast-path Verdict (`judge_strategy="primary_classifier"`), or: `on_escalation` ->
`on_cost_estimate` gate -> pre-flight cost guard -> `DebateEngine.debate` -> mid-flight
cost guard -> `judge.judge` -> Jury sets `was_escalated` / `persona_failures` and
backfills unset fields -> `on_verdict`.

## Commands

Python (from `packages/python`; venv already exists at `.venv`):
```
.venv/bin/python -m pytest tests -q        # full suite, under 1 s
.venv/bin/ruff check src tests
.venv/bin/black --check src tests
pip install -e ".[dev]"                     # fresh setup
```
TypeScript (from `packages/typescript`; deps hoisted to the root `node_modules`):
```
npm test          # node --test --experimental-strip-types, runs src directly
npm run check     # tsc build to dist + type-check examples against dist
npm run lint      # eslint
```
Root `npm test` runs both suites. Real-API smoke tests skip without `OPENAI_API_KEY`;
never add tests that need the network.

In a git worktree: the venv's editable install points at the main checkout, so run Python
with `PYTHONPATH=$PWD/src` and confirm `llm_jury.__file__` is in the worktree; symlink the
root `node_modules` into the worktree instead of installing.

Release: `gh workflow run release.yml -f version=X.Y.Z -f target=all|pypi|npm -f dry_run=false`.
It bumps `pyproject.toml`, `_version.py`, `package.json`, `_version.ts`, commits to main,
tags, tests, then publishes via OIDC trusted publishing. It commits and force-tags before
tests run, even on dry runs (REL-01 in `docs/REVIEW.md`).

Git: `main` has a required-review ruleset; self-authored PRs merge with
`gh pr merge --admin --squash`. No Claude attribution in commits or PR text.

## Conventions and gotchas

- Python is the source of truth. Behaviour changes land in both SDKs in the same PR, with
  snake_case names in Python and camelCase in TS. TS `Verdict.toDict()` uses camelCase
  keys; the TS CLI converts rows to snake_case so both CLIs emit the same JSONL.
- Escalation is strictly `confidence < threshold`; equal does not escalate.
- Failed persona calls (LLM error or unparseable output) stay in the transcript with
  `failed=True` and carry no vote. Majority, weighted and Bayesian judges use only the
  final round's valid responses; `LLMJudge` sees all rounds. An all-failed final round
  falls back to the primary result.
- Jury is authoritative for `was_escalated` and `persona_failures`. Judges return
  `total_duration_ms=0` and Jury fills it in.
- `judge_strategy` values are a public contract (README troubleshooting table lists them):
  `primary_classifier`, `majority_vote`, `weighted_vote`, `bayesian`, `llm_judge`,
  `llm_judge_fallback_*`, `cost_guard_pre_flight`, `cost_guard_primary_fallback`,
  `cost_guard_user_override`.
- `LiteLLMClient` omits `temperature` for reasoning models (`o1`, `o3`, `gpt-5` prefixes).
- The TS `LiteLLMClient` does not compute cost; only custom clients report `costUsd`.
- CLI defaults (`independent`, 1 round) differ from SDK defaults (`deliberation`, 2 rounds).
- TS `Jury` logs nothing unless given a logger (`logger: console`).
- TS sources import with `.ts` extensions and run under `--experimental-strip-types`
  (Node 22.6+). The npm `test` glob `tests/**/*.test.ts` is unquoted, so sh expands it one
  directory level deep; keep test files at `tests/<area>/<name>.test.ts`.
- `AUDIT.md`, cited by older docs, was never committed. `docs/REVIEW.md` explains the old
  IDs (B8, R7, S1, F7 and so on) and replaces it.
- Docs and comments: no em dashes, sentence-case headings, describe the code as it is.
