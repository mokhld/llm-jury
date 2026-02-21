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

test("deliberation mode early consensus", async () => {
  const llm = new FakeLLMClient({
    A: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "a", key_factors: ["x"] }) },
    B: { content: JSON.stringify({ label: "safe", confidence: 0.8, reasoning: "b", key_factors: ["y"] }) },
    C: { content: JSON.stringify({ label: "safe", confidence: 0.7, reasoning: "c", key_factors: ["z"] }) },
  });
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 3 }), llm);
  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);
  assert.equal(transcript.rounds.length, 2);
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
