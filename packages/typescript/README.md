# llm-jury (TypeScript)

TypeScript package for confidence-based escalation with multi-persona LLM debate and auditable verdicts.

Default LLM model: `gpt-5-mini` (overridable in classifier/personas/judge config).

## Install

```bash
npm install @llm-jury/core
```

## TypeScript API

```ts
import {
  FunctionClassifier,
  Jury,
  MajorityVoteJudge,
  PersonaRegistry,
} from "@llm-jury/core";

const classifier = new FunctionClassifier(
  () => ["safe", 0.62],
  ["safe", "unsafe"],
);

const jury = new Jury({
  classifier,
  personas: PersonaRegistry.contentModeration(),
  confidenceThreshold: 0.7,
  judge: new MajorityVoteJudge(),
});

const verdict = await jury.classify("borderline message");
console.log(verdict.label, verdict.confidence, verdict.wasEscalated);
```

`@llm-jury/core` is a library (not a hosted API service). Its default LLM client sends requests to `POST /chat/completions` on `OPENAI_BASE_URL` / `LITELLM_BASE_URL` / `https://api.openai.com/v1`.

## Prerequisites

- Node.js `22.6+` (`--experimental-strip-types` is used by the test command)
- `OPENAI_API_KEY` for real LLM calls

## Run Tests

```bash
npm test
```

## Run Real API Smoke Test

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" node --test --experimental-strip-types tests/smoke/real-api.test.ts
```

## CLI Smoke Test

Create `input.jsonl`:

```json
{"text":"hello","predicted_label":"safe","predicted_confidence":0.95}
{"text":"borderline text","predicted_label":"unsafe","predicted_confidence":0.96}
```

Create `calibration.jsonl`:

```json
{"text":"hello","label":"safe","predicted_label":"safe","predicted_confidence":0.95}
{"text":"borderline text","label":"unsafe","predicted_label":"unsafe","predicted_confidence":0.56}
```

Run classification:

```bash
npm run build
node dist/cli/main.js classify \
  --input input.jsonl \
  --output verdicts.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --judge-model gpt-5-mini \
  --persona-model gpt-5-mini \
  --threshold 0.7 \
  --labels safe,unsafe
```

Run calibration:

```bash
npm run build
node dist/cli/main.js calibrate \
  --input calibration.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --judge-model gpt-5-mini \
  --persona-model gpt-5-mini \
  --labels safe,unsafe
```

Calibration requires a ground-truth `label` field on every row.

Supported classifier specs: `function`, `llm:<model>`, `huggingface:<model>`.
