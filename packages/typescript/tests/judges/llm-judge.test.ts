import test from "node:test";
import assert from "node:assert/strict";

import { LLMJudge, sumCosts } from "../../src/judges/llmJudge.ts";
import type { DebateTranscript } from "../../src/debate/engine.ts";
import { FakeLLMClient } from "../helpers.ts";

const transcript: DebateTranscript = {
  inputText: "text",
  primaryResult: { label: "unknown", confidence: 0.3 },
  rounds: [[
    { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "harm", keyFactors: ["harm"] },
    { personaName: "B", label: "safe", confidence: 0.4, reasoning: "context", keyFactors: ["context"] },
  ]],
  durationMs: 10,
  totalTokens: 10,
  totalCostUsd: 0.001,
};

test("llm judge parses JSON", async () => {
  const llm = new FakeLLMClient({
    judge: {
      content: JSON.stringify({
        label: "unsafe",
        confidence: 0.81,
        reasoning: "Harm argument is stronger",
        key_agreements: ["ambiguous"],
        key_disagreements: ["intent"],
        decisive_factor: "targeted harm",
      }),
    },
  });

  const verdict = await new LLMJudge({ model: "judge", llmClient: llm }).judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.confidence, 0.81);
});

test("llm judge falls back to primary on invalid JSON", async () => {
  const llm = new FakeLLMClient({
    judge: { content: "not-json" },
  });
  const verdict = await new LLMJudge({ model: "judge", llmClient: llm }).judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unknown");
  assert.equal(verdict.confidence, 0.3);
});

test("sumCosts preserves null when both inputs are unknown", () => {
  assert.equal(sumCosts(null, null), null);
  assert.equal(sumCosts(undefined, undefined), null);
  assert.equal(sumCosts(null, undefined), null);
});

test("sumCosts treats null/undefined component as 0 when the other is known", () => {
  assert.equal(sumCosts(null, 0.5), 0.5);
  assert.equal(sumCosts(0.3, undefined), 0.3);
  assert.equal(sumCosts(0.5, 0.25), 0.75);
});

test("llm judge reports null totalCostUsd when neither transcript nor payload cost is known", async () => {
  const transcriptWithoutCost: DebateTranscript = { ...transcript, totalCostUsd: null };
  const llm = new FakeLLMClient({
    judge: {
      content: JSON.stringify({ label: "unsafe", confidence: 0.8, reasoning: "r" }),
      costUsd: undefined,
    },
  });
  // FakeLLMClient defaults costUsd to 0.001 when undefined, so override explicitly via direct fake.
  const nullCostClient = {
    async complete() {
      return { content: JSON.stringify({ label: "unsafe", confidence: 0.8, reasoning: "r" }), tokens: 10, costUsd: null as unknown as number };
    },
  };
  const verdict = await new LLMJudge({ model: "judge", llmClient: nullCostClient }).judge(transcriptWithoutCost, ["safe", "unsafe"]);
  assert.equal(verdict.totalCostUsd, null, "totalCostUsd preserved as null, not silently coerced to 0");
  // Sanity: the FakeLLMClient path with default costUsd should still produce a number.
  const verdict2 = await new LLMJudge({ model: "judge", llmClient: llm }).judge(transcript, ["safe", "unsafe"]);
  assert.equal(typeof verdict2.totalCostUsd, "number");
});
