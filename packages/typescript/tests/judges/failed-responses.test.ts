import test from "node:test";
import assert from "node:assert/strict";

import { BayesianJudge } from "../../src/judges/bayesian.ts";
import { LLMJudge } from "../../src/judges/llmJudge.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { WeightedVoteJudge } from "../../src/judges/weightedVote.ts";
import type { DebateTranscript } from "../../src/debate/engine.ts";
import type { PersonaResponse } from "../../src/personas/base.ts";
import { FakeLLMClient } from "../helpers.ts";

function failedResponse(personaName: string, label = "safe"): PersonaResponse {
  return {
    personaName,
    label,
    confidence: 0,
    reasoning: "Persona call failed: AuthenticationError: no api key",
    keyFactors: [],
    failed: true,
  };
}

function makeTranscript(rounds: PersonaResponse[][]): DebateTranscript {
  return {
    inputText: "text",
    primaryResult: { label: "unsafe", confidence: 0.55 },
    rounds,
    durationMs: 10,
    totalTokens: 20,
    totalCostUsd: 0.001,
  };
}

// Regression guard: a debate where every persona call failed (e.g. missing
// API key) used to yield a unanimous labels[0] verdict at confidence 1.0
// under majority vote, and the opposite label at ~1.0 under Bayesian.

test("majority vote: all-failed round falls back to primary result", async () => {
  const transcript = makeTranscript([[failedResponse("A"), failedResponse("B"), failedResponse("C")]]);
  const verdict = await new MajorityVoteJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.confidence, 0.55);
  assert.match(verdict.reasoning, /failed/);
});

test("majority vote: failed responses carry no vote", async () => {
  const transcript = makeTranscript([[
    { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "r1", keyFactors: [] },
    { personaName: "B", label: "unsafe", confidence: 0.8, reasoning: "r2", keyFactors: [] },
    failedResponse("C"),
  ]]);
  const verdict = await new MajorityVoteJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  // Confidence is computed over the two real votes, not three.
  assert.equal(verdict.confidence, 1);
});

test("majority vote: failed placeholders cannot flip the winner", async () => {
  const transcript = makeTranscript([[
    { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "r1", keyFactors: [] },
    failedResponse("B"),
    failedResponse("C"),
  ]]);
  const verdict = await new MajorityVoteJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
});

test("weighted vote: all-failed round falls back to primary result", async () => {
  const transcript = makeTranscript([[failedResponse("A"), failedResponse("B")]]);
  const verdict = await new WeightedVoteJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.confidence, 0.55);
});

test("bayesian: all-failed round falls back to primary result", async () => {
  const transcript = makeTranscript([[failedResponse("A"), failedResponse("B"), failedResponse("C")]]);
  const verdict = await new BayesianJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.confidence, 0.55);
});

test("bayesian: failed responses do not skew the posterior", async () => {
  const transcript = makeTranscript([[
    { personaName: "A", label: "safe", confidence: 0.9, reasoning: "r1", keyFactors: [] },
    failedResponse("B"),
  ]]);
  const verdict = await new BayesianJudge().judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "safe");
});

test("llm judge: skips the LLM call when every persona failed", async () => {
  const transcript = makeTranscript([[failedResponse("A"), failedResponse("B")]]);
  const client = new FakeLLMClient();
  const verdict = await new LLMJudge({ llmClient: client }).judge(transcript, ["safe", "unsafe"]);
  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.judgeStrategy, "llm_judge_fallback_personas_failed");
  assert.equal(client.calls.length, 0);
});

test("llm judge: prompt excludes failed responses", () => {
  const transcript = makeTranscript([[
    { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "real reasoning", keyFactors: [] },
    failedResponse("B"),
  ]]);
  const prompt = new LLMJudge({ llmClient: new FakeLLMClient() }).buildPrompt(transcript, ["safe", "unsafe"]);
  assert.match(prompt, /real reasoning/);
  assert.doesNotMatch(prompt, /Persona call failed/);
});
