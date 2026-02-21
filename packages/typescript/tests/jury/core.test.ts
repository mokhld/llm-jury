import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { Jury } from "../../src/jury/core.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { PersonaRegistry } from "../../src/personas/registry.ts";
import { FakeLLMClient } from "../helpers.ts";

test("fast path skips escalation", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.95], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });

  const verdict = await jury.classify("hello");

  assert.equal(verdict.wasEscalated, false);
  assert.equal(verdict.label, "safe");
  assert.equal(jury.stats.total, 1);
});

test("low confidence escalates", async () => {
  const classifier = new FunctionClassifier(() => ["unknown", 0.4], ["safe", "unsafe"]);
  const llm = new FakeLLMClient({
    "Policy Analyst": { content: JSON.stringify({ label: "unsafe", confidence: 0.9, reasoning: "harm", key_factors: ["harm"] }) },
    "Cultural Context Expert": { content: JSON.stringify({ label: "safe", confidence: 0.5, reasoning: "context", key_factors: ["context"] }) },
    "Harm Assessment Specialist": { content: JSON.stringify({ label: "unsafe", confidence: 0.8, reasoning: "risk", key_factors: ["risk"] }) },
  });

  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.contentModeration(),
    confidenceThreshold: 0.7,
    judge: new MajorityVoteJudge(),
    llmClient: llm,
  });

  const verdict = await jury.classify("ambiguous");

  assert.equal(verdict.wasEscalated, true);
  assert.equal(verdict.label, "unsafe");
  assert.equal(jury.stats.escalated, 1);
});

test("classifyBatch preserves input order", async () => {
  const classifier = new FunctionClassifier(async (text: string) => {
    if (text === "slow") {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    if (text === "fast") {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    return [text, 0.99];
  }, ["slow", "fast"]);

  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });
  const verdicts = await jury.classifyBatch(["slow", "fast"], 2);
  assert.deepEqual(
    verdicts.map((v) => v.label),
    ["slow", "fast"],
  );
});

test("escalation override can force fast-path", async () => {
  const classifier = new FunctionClassifier(() => ["unsafe", 0.1], ["safe", "unsafe"]);
  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.contentModeration(),
    escalationOverride: () => false,
  });
  const verdict = await jury.classify("text");
  assert.equal(verdict.wasEscalated, false);
});

test("maxDebateCostUsd triggers fallback", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const llm = new FakeLLMClient({
    "Policy Analyst": { content: JSON.stringify({ label: "unsafe", confidence: 0.9, reasoning: "harm", key_factors: ["harm"] }), costUsd: 0.5 },
    "Cultural Context Expert": { content: JSON.stringify({ label: "safe", confidence: 0.4, reasoning: "context", key_factors: ["context"] }), costUsd: 0.5 },
    "Harm Assessment Specialist": { content: JSON.stringify({ label: "unsafe", confidence: 0.8, reasoning: "risk", key_factors: ["risk"] }), costUsd: 0.5 },
  });
  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.contentModeration(),
    llmClient: llm,
    maxDebateCostUsd: 0.6,
  });

  const verdict = await jury.classify("ambiguous");
  assert.equal(verdict.judgeStrategy, "cost_guard_primary_fallback");
  assert.equal(verdict.label, "safe");
});

test("debateConcurrency is configurable", () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.95], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], debateConcurrency: 2 });
  assert.equal((jury.debateEngine as unknown as { concurrency: number }).concurrency, 2);
});
