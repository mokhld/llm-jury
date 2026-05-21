import test from "node:test";
import assert from "node:assert/strict";

import { LiteLLMClient, isRetryableError } from "../../src/llm/client.ts";

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

test("isRetryableError retries on status 429 / 5xx (property), rejects 4xx", () => {
  const e429 = Object.assign(new Error("rate limited"), { status: 429 });
  const e503 = Object.assign(new Error("unavailable"), { status: 503 });
  const e500 = Object.assign(new Error("server"), { statusCode: 500 });
  const e400 = Object.assign(new Error("bad request"), { status: 400 });
  const eResponse = Object.assign(new Error("nested"), { response: { status: 502 } });

  assert.equal(isRetryableError(e429), true);
  assert.equal(isRetryableError(e503), true);
  assert.equal(isRetryableError(e500), true);
  assert.equal(isRetryableError(eResponse), true);
  assert.equal(isRetryableError(e400), false);
});

test("isRetryableError falls back to message regex for back-compat", () => {
  assert.equal(isRetryableError(new Error("LLM request failed (503): unavailable")), true);
  assert.equal(isRetryableError(new Error("LLM request failed (400): bad")), false);
});

test("isRetryableError retries TypeError (fetch network failure) and AbortError", () => {
  const abort = new Error("aborted");
  abort.name = "AbortError";
  assert.equal(isRetryableError(new TypeError("fetch failed")), true);
  assert.equal(isRetryableError(abort), true);
});

test("litellm client retries 503 then succeeds", async () => {
  let attempts = 0;
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (_input: unknown, _init?: RequestInit) => {
    attempts += 1;
    if (attempts === 1) {
      return new Response("unavailable", { status: 503 });
    }
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: "{\"label\":\"safe\",\"confidence\":0.9}" } }],
        usage: { total_tokens: 5 },
      }),
      { status: 200 },
    );
  }) as typeof fetch;

  try {
    const client = new LiteLLMClient({ apiKey: "test-key", timeoutMs: 5000 });
    const result = await client.complete("gpt-4o-mini", "system", "prompt", 0);
    assert.equal(attempts, 2);
    assert.ok(result.content.includes("safe"));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("litellm client does NOT retry on 400", async () => {
  let attempts = 0;
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async () => {
    attempts += 1;
    return new Response("bad input", { status: 400 });
  }) as typeof fetch;

  try {
    const client = new LiteLLMClient({ apiKey: "test-key", timeoutMs: 5000 });
    await assert.rejects(client.complete("gpt-4o-mini", "system", "prompt", 0));
    assert.equal(attempts, 1, "400 must not retry");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
