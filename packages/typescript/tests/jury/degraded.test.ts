import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { Jury } from "../../src/jury/core.ts";
import type { Persona } from "../../src/personas/base.ts";

class AlwaysFailingClient {
  async complete(): Promise<{ content: string; tokens: number; costUsd: number }> {
    throw new Error("no api key");
  }
}

function makePersonas(): Persona[] {
  return ["A", "B", "C"].map((name) => ({
    name,
    role: "role",
    systemPrompt: name,
    model: "gpt-fake",
    temperature: 0,
  }));
}

// End-to-end regression for the README quickstart with a broken LLM setup.
// Before the failed-vote fix, a missing API key produced a fabricated
// unanimous labels[0] verdict at confidence 1.0 — silently flipping an
// "unsafe" primary classification to "safe".
test("all persona failures preserve the primary result", async () => {
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["unsafe", 0.55], ["safe", "unsafe"]),
    personas: makePersonas(),
    confidenceThreshold: 0.7,
    judge: new MajorityVoteJudge(),
    llmClient: new AlwaysFailingClient(),
  });

  const verdict = await jury.classify("borderline message");

  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.confidence, 0.55);
  assert.equal(verdict.wasEscalated, true);
  assert.equal(verdict.personaFailures, 3);
  assert.equal(verdict.debateDegraded, true);
});

test("healthy fast-path verdict is not degraded", async () => {
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["safe", 0.95], ["safe", "unsafe"]),
    personas: makePersonas(),
    confidenceThreshold: 0.7,
    judge: new MajorityVoteJudge(),
    llmClient: new AlwaysFailingClient(),
  });

  const verdict = await jury.classify("clearly fine");

  assert.equal(verdict.wasEscalated, false);
  assert.equal(verdict.personaFailures, 0);
  assert.equal(verdict.debateDegraded, false);
});

test("toDict includes degradation fields", async () => {
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["unsafe", 0.55], ["safe", "unsafe"]),
    personas: makePersonas(),
    confidenceThreshold: 0.7,
    judge: new MajorityVoteJudge(),
    llmClient: new AlwaysFailingClient(),
  });

  const verdict = await jury.classify("borderline message");
  const data = verdict.toDict();

  assert.equal(data.personaFailures, 3);
  assert.equal(data.debateDegraded, true);
});

function flakyJury(): Jury {
  return new Jury({
    classifier: new FunctionClassifier((text) => {
      if (text === "boom") {
        throw new Error("classifier exploded");
      }
      return ["safe", 0.95];
    }, ["safe", "unsafe"]),
    personas: [],
    confidenceThreshold: 0.7,
  });
}

test("classifyBatch rejects on first failure by default", async () => {
  await assert.rejects(
    () => flakyJury().classifyBatch(["ok", "boom", "ok"], 1),
    /classifier exploded/,
  );
});

test("classifyBatch with returnExceptions preserves successful verdicts", async () => {
  const results = await flakyJury().classifyBatch(["ok", "boom", "ok"], 1, true);

  assert.equal(results.length, 3);
  assert.equal((results[0] as { label: string }).label, "safe");
  assert.ok(results[1] instanceof Error);
  assert.equal((results[2] as { label: string }).label, "safe");
});
