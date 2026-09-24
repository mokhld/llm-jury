# Security Policy

## Supported versions

`llm-jury` is pre-1.0. Security fixes go to the latest `0.3.x` release,
and to `0.2.x` while users migrate:

| Package                        | Supported versions             |
| ------------------------------ | ------------------------------ |
| `llm-jury-classifier` (Python) | `0.3.x`, `0.2.x`               |
| `@llm-jury/core` (TypeScript)  | `0.3.x`, `0.2.x`               |

`0.1.x` no longer receives fixes. Upgrade to the latest published
version before reporting.

## Reporting a vulnerability

**Do not open a public GitHub issue for security problems.**

Email **emailmokhld@gmail.com** with:

- A description of the issue and the impact you believe it has.
- A minimal reproduction (code snippet, input, configuration).
- The affected SDK and version (`pip show llm-jury-classifier` /
  `npm ls @llm-jury/core`).
- Any suggested remediation if you have one.

You should expect an acknowledgement within **5 business days**. If you
do not, please follow up — your first mail may have been filtered.

## Disclosure process

1. We confirm the report and assess severity.
2. We develop and test a fix in a private branch.
3. We publish a patched release on PyPI / npm.
4. We publish a GitHub Security Advisory crediting the reporter (unless
   anonymity is requested).

We aim to ship a fix within **30 days** of confirmation for high /
critical severity issues. Lower-severity issues may be batched into the
next regular release.

## Scope

In scope:

- Code in `packages/python/src/llm_jury/` and
  `packages/typescript/src/`.
- Published packages on PyPI and npm.
- CI/CD workflows in `.github/workflows/`.

Out of scope:

- Vulnerabilities in upstream dependencies (report those to the
  dependency's own project; we will pick up patched versions via
  Dependabot once it is enabled, old audit item C2 in
  [docs/REVIEW.md](docs/REVIEW.md)).
- Prompt injection in user-supplied text. This is a known class of
  issue with LLM-based classification. The SDK fences the input in
  every prompt and validates returned labels; the README section
  [Prompt injection and untrusted input](README.md#prompt-injection-and-untrusted-input)
  describes what it does and what callers should still do. A bypass
  of the fencing or of label validation is in scope. There is no
  built-in injection detector (old audit item F11).
- Denial-of-wallet via large inputs. The SDK has no input length cap
  (old audit item S4), so callers are expected to enforce their own.
- Issues that require a malicious model provider or a compromised
  API key.

## Hardening recommendations for users

Even on a supported version you should:

- Set `max_debate_cost_usd` to bound spend, and set
  `estimated_cost_per_persona_usd` to a realistic per-call cost for your
  models, since calls that report no cost are charged at that estimate.
- Cap untrusted input length before passing it to a `Jury`.
- Route degraded and fallback verdicts (`debate_degraded`, or a
  `judge_strategy` of `cost_guard_*` or `llm_judge_fallback_*`) to human
  review.
- Treat persona prompts as untrusted output — don't `eval` or shell
  out to anything derived from a verdict.
- Rotate API keys regularly and scope them per-environment.
