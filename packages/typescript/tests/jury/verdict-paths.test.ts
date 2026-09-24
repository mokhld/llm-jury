import test from "node:test";
import assert from "node:assert/strict";

import type { ClassificationResult, Classifier } from "../../src/classifiers/base.ts";
import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { DebateConfig, DebateMode } from "../../src/debate/engine.ts";
import { Jury } from "../../src/jury/core.ts";
import type { JuryOptions } from "../../src/jury/core.ts";
import type { Verdict } from "../../src/judges/base.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { LiteLLMClient } from "../../src/llm/client.ts";
import type { Logger } from "../../src/logger.ts";
import type { Persona } from "../../src/personas/base.ts";
import { FakeLLMClient } from "../helpers.ts";

const labels = ["safe", "unsafe"];
const personas: Persona[] = [
  { name: "A", role: "r", systemPrompt: "A_PROMPT", model: "m", temperature: 0 },
  { name: "B", role: "r", systemPrompt: "B_PROMPT", model: "m", temperature: 0 },
];

class StaticClassifier implements Classifier {
  labels = labels;
  private result: ClassificationResult;
  constructor(result: ClassificationResult) {
    this.result = result;
  }
  async classify(): Promise<ClassificationResult> {
    return { ...this.result };
  }
}

async function verdictsSeen(options: Partial<JuryOptions>, confidence = 0.3): Promise<{ returned: Verdict; seen: Verdict[] }> {
  const seen: Verdict[] = [];
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["safe", confidence], labels),
    personas,
    judge: new MajorityVoteJudge(),
    llmClient: new FakeLLMClient(),
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
    onVerdict: (verdict) => seen.push(verdict),
    ...options,
  });
  const returned = await jury.classify("text");
  return { returned, seen };
}

test("onVerdict fires once for a fast-path verdict", async () => {
  const { returned, seen } = await verdictsSeen({}, 0.95);
  assert.equal(returned.judgeStrategy, "primary_classifier");
  assert.equal(seen.length, 1);
  assert.equal(seen[0], returned);
});

test("onVerdict fires once when onCostEstimate skips the debate", async () => {
  const { returned, seen } = await verdictsSeen({ onCostEstimate: () => false });
  assert.equal(returned.judgeStrategy, "cost_guard_user_override");
  assert.equal(seen.length, 1);
  assert.equal(seen[0], returned);
});

test("onVerdict fires once when the pre-flight estimate skips the debate", async () => {
  const { returned, seen } = await verdictsSeen({ maxDebateCostUsd: 0.001 });
  assert.equal(returned.judgeStrategy, "cost_guard_pre_flight");
  assert.equal(seen.length, 1);
  assert.equal(seen[0], returned);
});

test("onVerdict fires once when the debate runs over budget", async () => {
  const llm = new FakeLLMClient({
    A_PROMPT: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "r", key_factors: [] }), costUsd: 1 },
    B_PROMPT: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "r", key_factors: [] }), costUsd: 1 },
  });
  const { returned, seen } = await verdictsSeen({ llmClient: llm, maxDebateCostUsd: 0.5, estimatedCostPerPersonaUsd: 0.01 });
  assert.equal(returned.judgeStrategy, "cost_guard_primary_fallback");
  assert.equal(seen.length, 1);
  assert.equal(seen[0], returned);
});

test("onVerdict fires once for a judged verdict", async () => {
  const { returned, seen } = await verdictsSeen({});
  assert.equal(returned.judgeStrategy, "majority_vote");
  assert.equal(seen.length, 1);
  assert.equal(seen[0], returned);
});

test("a non-finite primary confidence escalates", async () => {
  for (const confidence of [NaN, Infinity, undefined, "0.99"]) {
    const llm = new FakeLLMClient();
    const jury = new Jury({
      classifier: new StaticClassifier({ label: "safe", confidence: confidence as number }),
      personas,
      judge: new MajorityVoteJudge(),
      llmClient: llm,
      confidenceThreshold: 0.7,
    });
    const verdict = await jury.classify("text");
    assert.equal(verdict.wasEscalated, true, `confidence ${String(confidence)} must escalate`);
    assert.ok(llm.calls.length > 0);
    // Prompts render the unusable primary confidence instead of crashing.
    assert.equal(verdict.personaFailures, 0);
    assert.match(llm.calls[0]!.prompt, /confidence: unknown/);
  }
});

test("escalationOverride still wins over a non-finite confidence", async () => {
  const jury = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: NaN }),
    personas,
    llmClient: new FakeLLMClient(),
    escalationOverride: () => false,
  });
  const verdict = await jury.classify("text");
  assert.equal(verdict.wasEscalated, false);
});

test("constructor rejects a confidenceThreshold outside [0, 1] or not finite", () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.9], labels);
  for (const bad of [NaN, Infinity, -0.1, 1.01, "0.5"]) {
    assert.throws(
      () => new Jury({ classifier, personas: [], confidenceThreshold: bad as number }),
      RangeError,
      `threshold ${String(bad)} must be rejected`,
    );
  }
  for (const ok of [0, 0.5, 1]) {
    assert.equal(new Jury({ classifier, personas: [], confidenceThreshold: ok }).threshold, ok);
  }
});

test("Jury passes its logger to the default LiteLLMClient", () => {
  const logger: Logger = { debug() {}, info() {}, warn() {}, error() {} };
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["safe", 0.9], labels),
    personas: [],
    logger,
  });
  const client = (jury.debateEngine as unknown as { llmClient: LiteLLMClient }).llmClient;
  assert.ok(client instanceof LiteLLMClient);
  assert.equal((client as unknown as { logger: Logger }).logger, logger);
});
