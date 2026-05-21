import test from "node:test";
import assert from "node:assert/strict";

import { DebateConfig, DebateEngine, DebateMode } from "../../src/debate/engine.ts";
import type { ClassificationResult } from "../../src/classifiers/base.ts";
import type { Persona } from "../../src/personas/base.ts";
import { buildPersonaResponseSchema } from "../../src/personas/schema.ts";
import { FakeLLMClient } from "../helpers.ts";

// Unique system prompts so FakeLLMClient's substring matcher routes cleanly —
// short single-letter tokens collide with words like "Available" in the prompt.
const personas: Persona[] = [
  { name: "Alpha", role: "role", systemPrompt: "ALPHA_PROMPT", model: "gpt-5-mini", temperature: 0.3 },
  { name: "Bravo", role: "role", systemPrompt: "BRAVO_PROMPT", model: "gpt-5-mini", temperature: 0.3 },
];
const primary: ClassificationResult = { label: "unknown", confidence: 0.3 };
const labels = ["safe", "unsafe"];

test("debate engine forwards response_format schema to llm client", async () => {
  const llm = new FakeLLMClient({
    ALPHA_PROMPT: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "a", key_factors: [], dissent_notes: null }) },
    BRAVO_PROMPT: { content: JSON.stringify({ label: "safe", confidence: 0.8, reasoning: "b", key_factors: [], dissent_notes: null }) },
  });
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.INDEPENDENT, maxRounds: 1 }), llm);

  await engine.debate("text", primary, labels);

  const expected = buildPersonaResponseSchema(labels);
  assert.ok(llm.calls.length >= 2);
  for (const call of llm.calls) {
    assert.deepEqual(call.responseFormat, expected);
  }
});

test("null dissent_notes from schema-enforced output maps to undefined", async () => {
  const llm = new FakeLLMClient({
    ALPHA_PROMPT: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "a", key_factors: [], dissent_notes: null }) },
    BRAVO_PROMPT: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "b", key_factors: [], dissent_notes: "actual rebuttal" }) },
  });
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.INDEPENDENT, maxRounds: 1 }), llm);

  const transcript = await engine.debate("text", primary, labels);
  const byName = Object.fromEntries(transcript.rounds[0]!.map((r) => [r.personaName, r]));

  assert.equal(byName.Alpha!.dissentNotes, undefined);
  assert.equal(byName.Bravo!.dissentNotes, "actual rebuttal");
});

test("LiteLLMClient body includes response_format when provided", async () => {
  const { LiteLLMClient } = await import("../../src/llm/client.ts");
  let body: Record<string, unknown> | null = null;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: JSON.stringify({ label: "safe", confidence: 0.9, reasoning: "x", key_factors: [], dissent_notes: null }) } }],
        usage: { total_tokens: 5 },
      }),
      { status: 200 },
    );
  }) as typeof fetch;

  try {
    const client = new LiteLLMClient({ apiKey: "test-key" });
    const rf = buildPersonaResponseSchema(labels) as unknown as Record<string, unknown>;
    await client.complete("gpt-4o-mini", "system", "prompt", 0, rf);
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.ok(body);
  assert.ok(body!.response_format, "response_format must be forwarded");
});
