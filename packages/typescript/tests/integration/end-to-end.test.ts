import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { Jury } from "../../src/jury/core.ts";
import { PersonaRegistry } from "../../src/personas/registry.ts";
import { LLMJudge } from "../../src/judges/llmJudge.ts";
import { FakeLLMClient } from "../helpers.ts";

test("end-to-end with mocked llm", async () => {
  const classifier = new FunctionClassifier(() => ["unknown", 0.45], ["safe", "unsafe"]);
  const llm = new FakeLLMClient({
    "Policy Analyst": { content: JSON.stringify({ label: "unsafe", confidence: 0.85, reasoning: "policy", key_factors: ["explicit content"] }) },
    "Cultural Context Expert": { content: JSON.stringify({ label: "safe", confidence: 0.6, reasoning: "satire", key_factors: ["context"] }) },
    "Harm Assessment Specialist": { content: JSON.stringify({ label: "unsafe", confidence: 0.75, reasoning: "potential harm", key_factors: ["vulnerable group"] }) },
    judge: {
      content: JSON.stringify({
        label: "unsafe",
        confidence: 0.8,
        reasoning: "Majority agrees and vulnerable group risk dominates",
        key_agreements: ["ambiguous text"],
        key_disagreements: ["satire context"],
        decisive_factor: "vulnerable group targeting",
      }),
    },
  });

  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.contentModeration(),
    confidenceThreshold: 0.7,
    judge: new LLMJudge({ model: "judge", llmClient: llm }),
    llmClient: llm,
  });

  const verdict = await jury.classify("some borderline content");

  assert.equal(verdict.wasEscalated, true);
  assert.equal(verdict.label, "unsafe");
  assert.ok(verdict.confidence > 0.7);
  assert.ok(verdict.reasoning.includes("vulnerable group"));
});
