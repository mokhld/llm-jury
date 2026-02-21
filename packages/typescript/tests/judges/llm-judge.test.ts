import test from "node:test";
import assert from "node:assert/strict";

import { LLMJudge } from "../../src/judges/llmJudge.ts";
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
