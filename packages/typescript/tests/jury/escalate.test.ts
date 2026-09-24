import test from "node:test";
import assert from "node:assert/strict";

import type { ClassificationResult } from "../../src/classifiers/base.ts";
import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { DebateConfig, DebateMode } from "../../src/debate/engine.ts";
import type { Verdict } from "../../src/judges/base.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { Jury } from "../../src/jury/core.ts";
import type { JuryOptions } from "../../src/jury/core.ts";
import { FakeLLMClient } from "../helpers.ts";

const labels = ["safe", "unsafe"];
const unsafeReply = {
  content: JSON.stringify({ label: "unsafe", confidence: 0.9, reasoning: "r", key_factors: [] }),
  costUsd: 0.001,
};

function makeJury(options: Partial<JuryOptions> = {}): { jury: Jury; client: FakeLLMClient; seen: unknown[] } {
  const seen: unknown[] = [];
  const client = new FakeLLMClient({ persona: unsafeReply });
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["safe", 0.4], labels),
    personas: ["A", "B"].map((name) => ({
      name,
      role: "r",
      systemPrompt: `persona-${name}`,
      model: "persona",
      temperature: 0,
    })),
    judge: new MajorityVoteJudge(),
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
    llmClient: client,
    onEscalation: (text) => seen.push(["escalation", text]),
    onVerdict: (verdict) => seen.push(["verdict", verdict]),
    ...options,
  });
  return { jury, client, seen };
}

test("escalate debates even a confident primary", async () => {
  const { jury, client } = makeJury();
  const primary: ClassificationResult = { label: "safe", confidence: 0.99, costUsd: 0 };
  const verdict = await jury.escalate("text", primary);

  assert.equal(verdict.wasEscalated, true);
  assert.equal(verdict.judgeStrategy, "majority_vote");
  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.primaryResult, primary);
  assert.equal(client.calls.length, 2);
  assert.ok(Math.abs((verdict.totalCostUsd ?? 0) - 0.002) < 1e-12);
});

test("escalate does not touch stats but fires callbacks", async () => {
  const { jury, seen } = makeJury();
  const verdict = await jury.escalate("text", { label: "safe", confidence: 0.5 });
  assert.deepEqual([jury.stats.total, jury.stats.fastPath, jury.stats.escalated], [0, 0, 0]);
  assert.deepEqual(seen, [
    ["escalation", "text"],
    ["verdict", verdict],
  ]);
});

test("escalate never calls the primary classifier", async () => {
  const { jury } = makeJury();
  const calls: string[] = [];
  jury.classifier.classify = async (text: string) => {
    calls.push(text);
    return { label: "safe", confidence: 0.4 };
  };
  await jury.escalate("text", { label: "safe", confidence: 0.4 });
  assert.deepEqual(calls, []);
});

test("escalate applies the cost gates", async () => {
  const override = makeJury({ onCostEstimate: () => false });
  const skipped = await override.jury.escalate("text", { label: "safe", confidence: 0.4 });
  assert.equal(skipped.judgeStrategy, "cost_guard_user_override");

  const capped = makeJury({ maxDebateCostUsd: 0.0001 });
  const preFlight = await capped.jury.escalate("text", { label: "safe", confidence: 0.4 });
  assert.equal(preFlight.judgeStrategy, "cost_guard_pre_flight");
  assert.equal(capped.client.calls.length, 0);
});

test("escalate matches the escalated branch of classify", async () => {
  const { jury } = makeJury();
  const viaClassify = await jury.classify("text");
  const primary = await jury.classifier.classify("text");
  const viaEscalate = await jury.escalate("text", primary);

  for (const field of ["label", "confidence", "judgeStrategy", "totalCostUsd", "personaFailures"] as const) {
    assert.equal(viaClassify[field], viaEscalate[field], field);
  }
  assert.equal(jury.stats.escalated, 1);
});

test("classify escalates through the shared branch", async () => {
  const { jury } = makeJury();
  const seen: string[] = [];
  const internals = jury as unknown as {
    runEscalation: (text: string, primary: ClassificationResult, start: number) => Promise<Verdict>;
  };
  const original = internals.runEscalation.bind(jury);
  internals.runEscalation = async (text, primary, start) => {
    seen.push(text);
    return original(text, primary, start);
  };
  await jury.classify("low");
  await jury.escalate("direct", { label: "safe", confidence: 0.9 });
  assert.deepEqual(seen, ["low", "direct"]);
});

test("escalate requires personas", async () => {
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["safe", 0.4], labels),
    personas: [],
    llmClient: new FakeLLMClient(),
  });
  await assert.rejects(jury.escalate("text", { label: "safe", confidence: 0.4 }), /at least one persona/);
});
