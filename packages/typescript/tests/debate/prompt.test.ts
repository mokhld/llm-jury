import test from "node:test";
import assert from "node:assert/strict";

import { DebateConfig, DebateEngine, DebateMode } from "../../src/debate/engine.ts";
import type { ClassificationResult } from "../../src/classifiers/base.ts";
import type { Persona, PersonaResponse } from "../../src/personas/base.ts";

const persona: Persona = {
  name: "Test Persona",
  role: "role",
  systemPrompt: "prompt",
  model: "gpt-5-mini",
  temperature: 0.3,
};

const primary: ClassificationResult = {
  label: "safe",
  confidence: 0.51,
};

test("prompt includes input and labels", () => {
  const engine = new DebateEngine([persona], new DebateConfig());
  const prompt = engine.buildPersonaPrompt(persona, "test input", primary, ["safe", "unsafe"], []);

  assert.ok(prompt.includes("test input"));
  assert.ok(prompt.includes("safe, unsafe"));
});

test("prompt includes prior rounds", () => {
  const engine = new DebateEngine([persona], new DebateConfig());
  const prior: PersonaResponse[][] = [[
    {
      personaName: "A",
      label: "safe",
      confidence: 0.8,
      reasoning: "context",
      keyFactors: ["context"],
    },
  ]];

  const prompt = engine.buildPersonaPrompt(persona, "test", primary, ["safe", "unsafe"], prior);
  assert.ok(prompt.includes("Round 1"));
  assert.ok(prompt.includes("context"));
});

test("adversarial prompt includes assigned role", () => {
  const engine = new DebateEngine([persona], new DebateConfig({ mode: DebateMode.ADVERSARIAL }));
  const prompt = engine.buildPersonaPrompt(persona, "test input", primary, ["safe", "unsafe"], []);
  assert.ok(prompt.includes("Adversarial Role"));
});

test("persona parse fallback includes persona name", () => {
  const engine = new DebateEngine([persona], new DebateConfig());
  const result = engine.parsePersonaResponse("not-json", "Policy Analyst");
  assert.equal(result.label, "unknown");
  assert.equal(result.confidence, 0);
  assert.ok(result.reasoning.length > 0);
});
