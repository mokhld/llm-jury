import test from "node:test";
import assert from "node:assert/strict";

import { BayesianJudge } from "../../src/judges/bayesian.ts";
import type { DebateTranscript } from "../../src/debate/engine.ts";

test("bayesian judge falls back when no responses", async () => {
  const transcript: DebateTranscript = {
    inputText: "text",
    primaryResult: { label: "safe", confidence: 0.77 },
    rounds: [[]],
    durationMs: 5,
    totalTokens: 0,
    totalCostUsd: 0,
  };

  const verdict = await new BayesianJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "safe");
  assert.equal(verdict.confidence, 0.77);
});

test("bayesian judge aggregates posteriors", async () => {
  const transcript: DebateTranscript = {
    inputText: "text",
    primaryResult: { label: "safe", confidence: 0.4 },
    rounds: [[
      { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "harm", keyFactors: ["harm"] },
      { personaName: "B", label: "unsafe", confidence: 0.8, reasoning: "risk", keyFactors: ["risk"] },
      { personaName: "C", label: "safe", confidence: 0.55, reasoning: "context", keyFactors: ["context"] },
    ]],
    durationMs: 5,
    totalTokens: 0,
    totalCostUsd: 0,
  };

  const judge = new BayesianJudge({
    A: { unsafe: 0.8, safe: 0.2 },
    B: { unsafe: 0.7, safe: 0.3 },
    C: { unsafe: 0.3, safe: 0.7 },
  });

  const verdict = await judge.judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  assert.ok(verdict.confidence > 0.5);
});
