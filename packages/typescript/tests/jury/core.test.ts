import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { DebateConfig, DebateMode } from "../../src/debate/engine.ts";
import { Jury } from "../../src/jury/core.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import type { Persona } from "../../src/personas/base.ts";
import { PersonaRegistry } from "../../src/personas/registry.ts";

const persona = (name: string): Persona => ({
  name,
  role: "r",
  systemPrompt: "s",
  model: "test-model",
  temperature: 0,
});
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

test("classifyBatch preserves order and runs each input exactly once under high concurrency", async () => {
  const inputs = Array.from({ length: 50 }, (_, i) => `item-${i}`);
  const callCounts = new Map<string, number>();
  let concurrentNow = 0;
  let peakConcurrency = 0;

  const classifier = new FunctionClassifier(async (text: string) => {
    callCounts.set(text, (callCounts.get(text) ?? 0) + 1);
    concurrentNow += 1;
    peakConcurrency = Math.max(peakConcurrency, concurrentNow);
    await new Promise((resolve) => setTimeout(resolve, Math.floor(Math.random() * 5)));
    concurrentNow -= 1;
    return [text, 0.99];
  }, inputs);

  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });
  const verdicts = await jury.classifyBatch(inputs, 8);

  assert.deepEqual(verdicts.map((v) => v.label), inputs, "output order must match input order");
  for (const text of inputs) {
    assert.equal(callCounts.get(text), 1, `each input classified exactly once (${text})`);
  }
  assert.ok(peakConcurrency <= 8, `peak concurrency (${peakConcurrency}) must not exceed limit`);
  assert.ok(peakConcurrency >= 2, `peak concurrency (${peakConcurrency}) should exceed 1 to prove parallelism`);
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

test("estimatedMaxDebateCostUsd matches N x rounds x per-persona", () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.95], ["safe", "unsafe"]);
  const personas = Array.from({ length: 3 }, (_, i) => persona(`P${i}`));
  const jury = new Jury({
    classifier,
    personas,
    debateConfig: new DebateConfig({ maxRounds: 2 }),
    estimatedCostPerPersonaUsd: 0.02,
  });
  assert.equal(jury.estimatedMaxDebateCostUsd, 0.12);
});

test("pre-flight estimate skips debate when over cap", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const llm = new FakeLLMClient();
  const personas = Array.from({ length: 4 }, (_, i) => persona(`P${i}`));
  const jury = new Jury({
    classifier,
    personas,
    llmClient: llm,
    maxDebateCostUsd: 0.05,
    estimatedCostPerPersonaUsd: 0.01,
    debateConfig: new DebateConfig({ maxRounds: 2 }),
  });

  // 4 × 2 × 0.01 = 0.08 > 0.05 → pre-flight refusal
  const verdict = await jury.classify("text");
  assert.equal(verdict.judgeStrategy, "cost_guard_pre_flight");
  assert.equal(verdict.label, "safe");
  assert.equal(verdict.wasEscalated, true);
  assert.equal(verdict.debateTranscript, null);
  assert.equal(llm.calls.length, 0, "no LLM calls when pre-flight trips");
});

test("per-batch cost guard halts new batches once cap exceeded", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const personas = Array.from({ length: 6 }, (_, i) => persona(`P${i}`));

  // Each persona call costs 0.10. With concurrency=2 and cap=0.05, after batch 1
  // (2 calls, cumulative=0.20) the guard kicks in and skips batches 2 & 3.
  const replies: Record<string, { content: string; costUsd: number }> = {};
  for (const persona of personas) {
    replies[persona.name] = {
      content: JSON.stringify({ label: "safe", confidence: 0.7, reasoning: "r", key_factors: [] }),
      costUsd: 0.10,
    };
  }
  const llm = new FakeLLMClient(replies);

  const jury = new Jury({
    classifier,
    personas,
    llmClient: llm,
    debateConcurrency: 2,
    maxDebateCostUsd: 0.05,
    estimatedCostPerPersonaUsd: 0.001, // keep pre-flight quiet so we exercise the batch guard
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT, maxRounds: 1 }),
  });

  await jury.classify("text");
  assert.equal(llm.calls.length, 2, "second and third batches must be skipped after cap is exceeded");
});

test("F4: onCostEstimate returning false skips the debate", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const llm = new FakeLLMClient();
  const personas = [persona("A")];
  const received: Array<{ estimate: number; text: string }> = [];

  const jury = new Jury({
    classifier,
    personas,
    llmClient: llm,
    onCostEstimate: (estimate, text) => {
      received.push({ estimate, text });
      return false;
    },
  });

  const verdict = await jury.classify("hello");

  assert.equal(verdict.judgeStrategy, "cost_guard_user_override");
  assert.equal(verdict.wasEscalated, true);
  assert.equal(verdict.label, "safe");
  assert.equal(verdict.debateTranscript, null);
  assert.equal(llm.calls.length, 0, "no LLM calls when callback skips");
  assert.equal(received.length, 1);
  assert.equal(received[0]!.estimate, jury.estimatedMaxDebateCostUsd);
  assert.equal(received[0]!.text, "hello");
});

test("F4: onCostEstimate returning true proceeds with the debate", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const llm = new FakeLLMClient();
  const personas = [persona("A")];
  const jury = new Jury({
    classifier,
    personas,
    llmClient: llm,
    onCostEstimate: () => true,
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT, maxRounds: 1 }),
  });

  const verdict = await jury.classify("hello");
  assert.notEqual(verdict.judgeStrategy, "cost_guard_user_override");
  assert.ok(llm.calls.length > 0, "debate ran");
});

test("F4: onCostEstimate returning undefined proceeds", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const llm = new FakeLLMClient();
  const personas = [persona("A")];
  const jury = new Jury({
    classifier,
    personas,
    llmClient: llm,
    onCostEstimate: () => undefined,
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT, maxRounds: 1 }),
  });

  const verdict = await jury.classify("hello");
  assert.notEqual(verdict.judgeStrategy, "cost_guard_user_override");
  assert.ok(llm.calls.length > 0);
});

test("F4: onCostEstimate fires before maxDebateCostUsd guard", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const personas = Array.from({ length: 4 }, (_, i) => persona(`P${i}`));
  const llm = new FakeLLMClient();

  const jury = new Jury({
    classifier,
    personas,
    llmClient: llm,
    maxDebateCostUsd: 0.05,
    estimatedCostPerPersonaUsd: 0.01, // 4 × 2 × 0.01 = 0.08 > 0.05
    onCostEstimate: () => false,
  });

  // Without the callback this would be cost_guard_pre_flight.
  const verdict = await jury.classify("text");
  assert.equal(verdict.judgeStrategy, "cost_guard_user_override");
});
