import test from "node:test";
import assert from "node:assert/strict";

import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { WeightedVoteJudge } from "../../src/judges/weightedVote.ts";
import type { DebateTranscript } from "../../src/debate/engine.ts";

const transcript: DebateTranscript = {
  inputText: "text",
  primaryResult: { label: "unknown", confidence: 0.3 },
  rounds: [[
    { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "harm", keyFactors: ["harm"] },
    { personaName: "B", label: "safe", confidence: 0.4, reasoning: "context", keyFactors: ["context"] },
    { personaName: "C", label: "unsafe", confidence: 0.7, reasoning: "risk", keyFactors: ["risk"] },
  ]],
  durationMs: 10,
  totalTokens: 10,
  totalCostUsd: 0.001,
};

test("majority vote", async () => {
  const verdict = await new MajorityVoteJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  assert.ok(verdict.confidence > 0.6);
});

test("weighted vote", async () => {
  const verdict = await new WeightedVoteJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
});
