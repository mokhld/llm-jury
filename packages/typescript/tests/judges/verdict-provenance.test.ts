import test from "node:test";
import assert from "node:assert/strict";

import { LIBRARY_VERSION } from "../../src/_version.ts";
import type { ClassificationResult } from "../../src/classifiers/base.ts";
import type { DebateTranscript } from "../../src/debate/engine.ts";
import { Verdict } from "../../src/judges/base.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";

const ISO_8601 = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/;

function buildTranscript(): DebateTranscript {
  const primary: ClassificationResult = { label: "unknown", confidence: 0.4 };
  return {
    inputText: "text",
    primaryResult: primary,
    rounds: [[
      { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "r1", keyFactors: ["a"] },
      { personaName: "B", label: "unsafe", confidence: 0.8, reasoning: "r2", keyFactors: ["b"] },
      { personaName: "C", label: "unsafe", confidence: 0.7, reasoning: "r3", keyFactors: ["c"] },
    ]],
    durationMs: 10,
    totalTokens: 20,
    totalCostUsd: 0.001,
  };
}

test("verdict includes library version", async () => {
  const verdict = await new MajorityVoteJudge().judge(buildTranscript(), ["safe", "unsafe"]);
  assert.equal(verdict.libraryVersion, LIBRARY_VERSION);
});

test("verdict includes ISO-8601 createdAt", async () => {
  const verdict = await new MajorityVoteJudge().judge(buildTranscript(), ["safe", "unsafe"]);
  assert.ok(ISO_8601.test(verdict.createdAt), `createdAt should match ISO-8601, got ${verdict.createdAt}`);
});

test("verdict is an instance of Verdict (class, not plain object)", async () => {
  const verdict = await new MajorityVoteJudge().judge(buildTranscript(), ["safe", "unsafe"]);
  assert.ok(verdict instanceof Verdict);
});

test("verdict.toDict() includes provenance fields", async () => {
  const verdict = await new MajorityVoteJudge().judge(buildTranscript(), ["safe", "unsafe"]);
  const dict = verdict.toDict();
  assert.equal(dict.libraryVersion, LIBRARY_VERSION);
  assert.ok(ISO_8601.test(String(dict.createdAt)));
  assert.equal(dict.label, verdict.label);
});

test("JSON.stringify(verdict) serialises provenance via toJSON()", async () => {
  const verdict = await new MajorityVoteJudge().judge(buildTranscript(), ["safe", "unsafe"]);
  const payload = JSON.parse(JSON.stringify(verdict));
  assert.equal(payload.libraryVersion, LIBRARY_VERSION);
  assert.ok(ISO_8601.test(payload.createdAt));
});

test("explicit init overrides default provenance", () => {
  const verdict = new Verdict({
    label: "x",
    confidence: 0.5,
    reasoning: "r",
    wasEscalated: false,
    primaryResult: { label: "x", confidence: 0.5 },
    debateTranscript: null,
    judgeStrategy: "test",
    totalDurationMs: 1,
    totalCostUsd: null,
    libraryVersion: "9.9.9",
    createdAt: "2026-01-01T00:00:00.000Z",
  });
  assert.equal(verdict.libraryVersion, "9.9.9");
  assert.equal(verdict.createdAt, "2026-01-01T00:00:00.000Z");
});
