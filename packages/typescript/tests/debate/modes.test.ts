import test from "node:test";
import assert from "node:assert/strict";

import { DebateConfig, DebateEngine, DebateMode } from "../../src/debate/engine.ts";
import type { ClassificationResult } from "../../src/classifiers/base.ts";
import type { Persona } from "../../src/personas/base.ts";
import { FakeLLMClient } from "../helpers.ts";

const personas: Persona[] = [
  { name: "A", role: "role", systemPrompt: "A", model: "gpt-5-mini", temperature: 0.3 },
  { name: "B", role: "role", systemPrompt: "B", model: "gpt-5-mini", temperature: 0.3 },
  { name: "C", role: "role", systemPrompt: "C", model: "gpt-5-mini", temperature: 0.3 },
];

const primary: ClassificationResult = { label: "unknown", confidence: 0.4 };

test("independent mode one round", async () => {
  const llm = new FakeLLMClient();
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.INDEPENDENT }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  assert.equal(transcript.rounds.length, 1);
  assert.equal(transcript.rounds[0].length, 3);
});

test("sequential mode one round", async () => {
  const llm = new FakeLLMClient();
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.SEQUENTIAL }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  assert.equal(transcript.rounds.length, 1);
});

test("deliberation mode stops after a unanimous opening round", async () => {
  const llm = new FakeLLMClient({
    A: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "a", key_factors: ["x"] }) },
    B: { content: JSON.stringify({ label: "safe", confidence: 0.8, reasoning: "b", key_factors: ["y"] }) },
    C: { content: JSON.stringify({ label: "safe", confidence: 0.7, reasoning: "c", key_factors: ["z"] }) },
  });
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 3 }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  // Consensus is checked after the opening round too: no deliberation
  // rounds and no summariser call, so only the 3 opening persona calls.
  assert.equal(transcript.rounds.length, 1);
  assert.equal(transcript.summary, undefined);
  assert.equal(llm.calls.length, 3);
});

test("F7: early stops on high confidence even with split labels", async () => {
  const llm = new FakeLLMClient({
    A: { content: JSON.stringify({ label: "safe", confidence: 0.97, reasoning: "a", key_factors: [] }) },
    B: { content: JSON.stringify({ label: "unsafe", confidence: 0.96, reasoning: "b", key_factors: [] }) },
    C: { content: JSON.stringify({ label: "safe", confidence: 0.98, reasoning: "c", key_factors: [] }) },
  });
  const engine = new DebateEngine(
    personas,
    new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 5, earlyStopMinConfidence: 0.95 }),
    llm,
  );
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  // The earlyStopMinConfidence rule is met by the opening round, so the
  // debate stops there: 1 round, not 5, and no summary.
  assert.equal(transcript.rounds.length, 1);
  assert.equal(transcript.summary, undefined);
});

// Opening round splits, first deliberation round is unanimous: the loop
// stops at that round and the summariser still runs.
class ConvergingClient {
  personaCalls = 0;
  summariserCalls = 0;
  async complete(_model: string, systemPrompt: string, prompt: string, _temperature = 0) {
    if (systemPrompt.startsWith("You are a neutral summarisation agent")) {
      this.summariserCalls += 1;
      return { content: "summary", tokens: 1, costUsd: 0.001 };
    }
    this.personaCalls += 1;
    const deliberating = prompt.includes("## Deliberation Instructions");
    const label = deliberating || systemPrompt === "A" ? "safe" : "unsafe";
    return {
      content: JSON.stringify({ label, confidence: 0.8, reasoning: "r", key_factors: [] }),
      tokens: 1,
      costUsd: 0.001,
    };
  }
}

test("deliberation consensus in a later round stops the loop and still summarises", async () => {
  const llm = new ConvergingClient();
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 4 }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  assert.equal(transcript.rounds.length, 2);
  assert.equal(llm.personaCalls, 6);
  assert.equal(llm.summariserCalls, 1);
  assert.equal(transcript.summary, "summary");
});

test("F7: does not early stop when one persona is below threshold", async () => {
  const llm = new FakeLLMClient({
    A: { content: JSON.stringify({ label: "safe", confidence: 0.97, reasoning: "a", key_factors: [] }) },
    B: { content: JSON.stringify({ label: "unsafe", confidence: 0.50, reasoning: "b", key_factors: [] }) },
    C: { content: JSON.stringify({ label: "safe", confidence: 0.96, reasoning: "c", key_factors: [] }) },
  });
  const engine = new DebateEngine(
    personas,
    new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 3, earlyStopMinConfidence: 0.95 }),
    llm,
  );
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  // Threshold not met → loop runs to maxRounds.
  assert.equal(transcript.rounds.length, 3);
});

test("F7: default preserves unanimous-only consensus (no early stop on split labels)", async () => {
  const llm = new FakeLLMClient({
    A: { content: JSON.stringify({ label: "safe", confidence: 0.99, reasoning: "a", key_factors: [] }) },
    B: { content: JSON.stringify({ label: "unsafe", confidence: 0.99, reasoning: "b", key_factors: [] }) },
    C: { content: JSON.stringify({ label: "safe", confidence: 0.99, reasoning: "c", key_factors: [] }) },
  });
  const engine = new DebateEngine(
    personas,
    new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 3 }),
    llm,
  );
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  assert.equal(transcript.rounds.length, 3);
});

test("adversarial mode assigns prosecution and defense roles", async () => {
  const llm = new FakeLLMClient();
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.ADVERSARIAL }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  assert.equal(transcript.rounds.length, 1);
  assert.equal(transcript.rounds[0].length, 3);

  const promptA = engine.buildPersonaPrompt(personas[0], "text", primary, ["safe", "unsafe"], []);
  const promptB = engine.buildPersonaPrompt(personas[1], "text", primary, ["safe", "unsafe"], []);
  assert.match(promptA, /Prosecution/);
  assert.match(promptB, /Defense/);
});

class FlakyLLMClient {
  failFor: Set<string>;
  constructor(failFor: Set<string>) {
    this.failFor = failFor;
  }
  async complete(_model: string, systemPrompt: string, _prompt: string, _temperature = 0) {
    if (this.failFor.has(systemPrompt)) {
      throw new Error(`simulated upstream failure for persona ${systemPrompt}`);
    }
    return {
      content: JSON.stringify({ label: "safe", confidence: 0.8, reasoning: "ok", key_factors: ["k"] }),
      tokens: 10,
      costUsd: 0.001,
    };
  }
}

test("one persona failure does not crash independent mode", async () => {
  const llm = new FlakyLLMClient(new Set(["B"]));
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.INDEPENDENT }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);

  assert.equal(transcript.rounds[0].length, 3);
  assert.deepEqual(transcript.rounds[0].map((r) => r.personaName), ["A", "B", "C"]);
  const failed = transcript.rounds[0].find((r) => r.personaName === "B")!;
  assert.equal(failed.confidence, 0);
  assert.match(failed.reasoning, /Persona call failed/);
});

test("one persona failure does not crash sequential mode", async () => {
  const llm = new FlakyLLMClient(new Set(["B"]));
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.SEQUENTIAL }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);

  assert.equal(transcript.rounds[0].length, 3);
  const failed = transcript.rounds[0].find((r) => r.personaName === "B")!;
  assert.match(failed.reasoning, /Persona call failed/);
});

test("one persona failure does not crash deliberation mode", async () => {
  const llm = new FlakyLLMClient(new Set(["B"]));
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 2 }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);

  assert.equal(transcript.rounds[0].length, 3);
  const failed = transcript.rounds[0].find((r) => r.personaName === "B")!;
  assert.equal(failed.confidence, 0);
  assert.match(failed.reasoning, /Persona call failed/);
});

// T9: single-persona deliberation. A one-persona round trivially has one
// unique label, so consensus is reached in the opening round and the
// debate stops there without deliberation rounds or a summary.
test("single-persona deliberation reaches consensus in the opening round", async () => {
  const singlePersona: Persona[] = [
    { name: "SOLO_PERSONA_TOKEN", role: "role", systemPrompt: "SOLO_PERSONA_TOKEN", model: "gpt-5-mini", temperature: 0.3 },
  ];
  const llm = new FakeLLMClient({
    SOLO_PERSONA_TOKEN: {
      content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "ok", key_factors: ["x"] }),
    },
  });
  const engine = new DebateEngine(
    singlePersona,
    new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 2 }),
    llm,
  );

  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);

  assert.equal(transcript.rounds.length, 1);
  assert.equal(transcript.rounds[0]!.length, 1);
  assert.equal(transcript.rounds[0]![0]!.label, "safe");
  assert.equal(transcript.summary, undefined, "consensus in the opening round skips the summariser");
  assert.equal(llm.calls.length, 1);
  assert.equal(engine.consensusReached(transcript.rounds[0]!), true);
});

// T9: no-response rounds — consensusReached([]) is defended at engine.ts:502
// but never exercised. Lock the behaviour so a future refactor that flips the
// empty-set semantics is a deliberate choice rather than a silent change.
test("consensusReached returns false for an empty round (no-response branch)", () => {
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.DELIBERATION }), new FakeLLMClient());
  assert.equal(engine.consensusReached([]), false);
});

// T5: summariser failure must not crash the verdict — persona rounds are
// load-bearing, the synthesis is best-effort.
class SummariserFailureClient {
  personaCalls = 0;
  summariserCalls = 0;
  async complete(_model: string, systemPrompt: string, _prompt: string, _temperature = 0) {
    if (systemPrompt.startsWith("You are a neutral summarisation agent")) {
      this.summariserCalls += 1;
      throw new Error("summariser is unavailable");
    }
    this.personaCalls += 1;
    // Split labels so there is no consensus and the summariser runs.
    const label = systemPrompt === "A" ? "safe" : "unsafe";
    return {
      content: JSON.stringify({ label, confidence: 0.8, reasoning: "ok", key_factors: ["k"] }),
      tokens: 10,
      costUsd: 0.001,
    };
  }
}

test("debate completes when summariser raises (T5)", async () => {
  const llm = new SummariserFailureClient();
  const engine = new DebateEngine(
    personas,
    new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 2 }),
    llm,
  );

  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);

  assert.equal(transcript.rounds.length, 2);
  for (const round of transcript.rounds) {
    assert.equal(round.length, 3);
  }
  assert.equal(llm.summariserCalls, 1, "summariser was invoked and failed");
  assert.equal(transcript.summary, undefined, "summary must be absent, not crashy");
});
