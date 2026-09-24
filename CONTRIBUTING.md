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

# Python (dev tools such as pytest are the "dev" extra)
cd packages/python
uv sync --extra dev   # or: pip install -e ".[dev]"
uv run python -m pytest -q

# TypeScript (the root package-lock.json covers the workspace)
cd ../..
npm ci
cd packages/typescript
npm run check     # tsc + examples type-check
npm test
```

## Project layout

```
packages/python/        llm-jury-classifier (source of truth)
packages/typescript/    @llm-jury/core (parity port)
examples/*.py           runnable Python examples
examples/typescript/    runnable TS examples (type-checked in CI)
docs/REVIEW.md          known problems and their status
docs/FEATURES.md        feature backlog with implementation briefs
CHANGELOG.md            user-facing change log
```

The Python package is the source of truth. When changing behaviour,
land Python first or in parallel, and mirror in TypeScript in the same
PR unless the change is genuinely SDK-specific.

## Running tests

```bash
# Python: 280+ tests, about 1 s
cd packages/python && uv run python -m pytest -q

# TypeScript: 230+ tests under node:test
cd packages/typescript && npm test

# TypeScript type-check + examples gate (CI runs this)
cd packages/typescript && npm run check
```

CI runs both suites on every PR across Python 3.10 to 3.13 and Node 22 and 24. It also builds the
wheel and the npm tarball, installs each into a clean environment and runs the installed
`llm-jury` command (the tarball on Node 20 and 22).

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

[`docs/REVIEW.md`](docs/REVIEW.md) is the source of truth for known
problems and their status. [`docs/FEATURES.md`](docs/FEATURES.md) ranks
the feature backlog and has a brief for each item that is detailed
enough to implement from.

Larger feature work (streaming, observability hooks, review routing,
prompt-injection detection) is open for discussion, but please open an
issue first.

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
