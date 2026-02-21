import test from "node:test";
import assert from "node:assert/strict";

import { LiteLLMClient } from "../../src/llm/client.ts";

test("litellm client omits temperature for gpt-5 models", async () => {
  let body: Record<string, unknown> | null = null;
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: "{\"label\":\"safe\",\"confidence\":0.9}" } }],
        usage: { total_tokens: 5 },
      }),
      { status: 200 },
    );
  }) as typeof fetch;

  try {
    const client = new LiteLLMClient({ apiKey: "test-key" });
    await client.complete("gpt-5-mini", "system", "prompt", 0.3);
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.ok(body);
  assert.equal(Object.prototype.hasOwnProperty.call(body, "temperature"), false);
});

test("litellm client includes temperature for non gpt-5 models", async () => {
  let body: Record<string, unknown> | null = null;
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: "{\"label\":\"safe\",\"confidence\":0.9}" } }],
        usage: { total_tokens: 5 },
      }),
      { status: 200 },
    );
  }) as typeof fetch;

  try {
    const client = new LiteLLMClient({ apiKey: "test-key" });
    await client.complete("gpt-4o-mini", "system", "prompt", 0.3);
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.ok(body);
  assert.equal(body?.temperature, 0.3);
});
