# Security Policy

## Supported versions

`llm-jury` is pre-1.0. Only the latest minor release of each SDK
receives security fixes:

| Package                  | Supported version |
| ------------------------ | ----------------- |
| `llm-jury-classifier` (Python) | latest `0.1.x`    |
| `@llm-jury/core` (TypeScript)  | latest `0.1.x`    |

Older versions will not receive backports. Upgrade to the latest
published version before reporting.

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
  Dependabot once it is enabled — see audit item C2).
- Prompt-injection in user-supplied text. This is a known class of
  issue with LLM-based classification — see audit items S1 / F11.
  Mitigation guidance is in the README; a built-in detector is on the
  roadmap as F11.
- Denial-of-wallet via large inputs. Tracked as audit item S4. Until
  it lands, callers are expected to enforce their own input caps.
- Issues that require a malicious model provider or a compromised
  API key.

## Hardening recommendations for users

Even on a supported version you should:

- Set `max_debate_cost_usd` to bound spend.
- Cap untrusted input length before passing it to a `Jury`.
- Treat persona prompts as untrusted output — don't `eval` or shell
  out to anything derived from a verdict.
- Rotate API keys regularly and scope them per-environment.
