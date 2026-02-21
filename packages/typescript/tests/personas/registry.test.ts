import test from "node:test";
import assert from "node:assert/strict";

import { PersonaRegistry } from "../../src/personas/registry.ts";

test("content moderation personas", () => {
  const personas = PersonaRegistry.contentModeration();
  const names = new Set(personas.map((p) => p.name));
  assert.ok(names.has("Policy Analyst"));
  assert.ok(names.has("Cultural Context Expert"));
  assert.ok(names.has("Harm Assessment Specialist"));
  assert.ok(personas.every((p) => Boolean(p.systemPrompt)));
});

test("custom personas", () => {
  const personas = PersonaRegistry.custom([
    {
      name: "Custom",
      role: "role",
      systemPrompt: "prompt",
      model: "gpt-5-mini",
    },
  ]);
  assert.equal(personas[0].name, "Custom");
});
