# Contributing to llm-jury

Thanks for considering a contribution. This is a dual-SDK monorepo —
**Python** (`packages/python`, published as `llm-jury-classifier`) and
**TypeScript** (`packages/typescript`, published as `@llm-jury/core`).
The two SDKs are intended to be a parity port, so most changes need to
land in both.

## Quick start

```bash
git clone https://github.com/mokhld/llm-jury.git
cd llm-jury

# Python
cd packages/python
uv sync           # or: pip install -e ".[dev]"
uv run python -m pytest -q

# TypeScript
cd ../typescript
npm install
npm run check     # tsc + examples type-check
npm test
```

## Project layout

```
packages/python/        — llm-jury-classifier (source of truth)
packages/typescript/    — @llm-jury/core (parity port)
examples/python/        — runnable Python examples
examples/typescript/    — runnable TS examples (type-checked in CI)
AUDIT.md                — open issues, prioritised roadmap, what landed
CHANGELOG.md            — user-facing change log
```

The Python package is the source of truth. When changing behaviour,
land Python first or in parallel, and mirror in TypeScript in the same
PR unless the change is genuinely SDK-specific.

## Running tests

```bash
# Python — 100+ tests, runs in <1s
cd packages/python && uv run python -m pytest -q

# TypeScript — 80+ tests under node:test
cd packages/typescript && npm test

# TypeScript type-check + examples gate (CI runs this)
cd packages/typescript && npm run check
```

CI runs both suites on every PR across Python 3.10–3.13 + Node 22.

## Linting

Lint is enforced in CI (one dedicated `lint` job, separate from the
test matrix):

```bash
# Python — ruff (lint) + black (format check)
cd packages/python
uv run ruff check src tests
uv run black --check src tests   # drop --check to auto-format

# TypeScript — eslint
cd packages/typescript
npm run lint
```

Configs live in `packages/python/pyproject.toml` (`[tool.ruff]`,
`[tool.black]`) and `packages/typescript/eslint.config.js`. If a
lint rule fights a deliberate pattern, prefer adjusting the config
over sprinkling `# noqa` / `eslint-disable` comments — and call
out the change in the PR.

## Workflow

1. Open an issue first for non-trivial work so we can align on scope.
2. Branch from `main`: `git checkout -b feat/short-description`.
3. Write tests. Every behaviour change needs a test that pins it.
4. Make sure both test suites pass locally before pushing.
5. Push and open a PR. Keep PRs focused — one concern per PR.
6. CI must be green. PRs are squash-merged for a clean history.

## Code style

Lint is enforced in CI (see the [Linting](#linting) section above).
House style on top of what the linters check:

- **Python**: `from __future__ import annotations` at the top of new
  modules. Tests use `unittest.IsolatedAsyncioTestCase`.
- **TypeScript**: explicit types on public APIs. Imports use `.ts`
  extensions (Node 22 ESM convention).

## Commit messages

Single commit per PR (we squash on merge). Format:

```
<area>: <short summary>

<optional longer explanation — what changed and why>
```

Areas in use: `feat`, `fix`, `tests`, `docs`, `ci`, `parity`, `chore`.
See `git log --oneline main` for examples.

## What to work on

`AUDIT.md` is the source of truth for what's open. Section 9 lists
the prioritised roadmap. Test-coverage gaps (T-rows in §7) are good
first contributions — small, mechanical, each one pins a real branch
in the code.

Larger feature work (P3 stretch items in §9: streaming, replay,
webhooks, OTel, audit-log export, prompt-injection detection) is open
for discussion but please open an issue first.

## Reporting bugs

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Include:

- which SDK (Python / TypeScript / both)
- minimal repro
- `llm-jury-classifier` or `@llm-jury/core` version
- Python / Node version
- stack trace if applicable

## Security

Don't open a public issue for security problems. See `SECURITY.md`
for the disclosure process.

## License

By contributing, you agree your contributions will be licensed under
the MIT license (see `LICENSE`).
