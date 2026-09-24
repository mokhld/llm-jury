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

test("isRetryableError ignores status-like numbers in the message when a structured status exists", () => {
  const e400 = Object.assign(new Error("LLM request failed (400): max_tokens must be <= 512"), { status: 400 });
  const e401 = Object.assign(new Error("invalid key 429abc 503"), { statusCode: 401 });
  assert.equal(isRetryableError(e400), false);
  assert.equal(isRetryableError(e401), false);
});

function okResponse(): Response {
  return new Response(
    JSON.stringify({
      choices: [{ message: { content: "{\"label\":\"safe\",\"confidence\":0.9}" } }],
      usage: { total_tokens: 5 },
    }),
    { status: 200 },
  );
}

/**
 * Run `fn` with fetch returning `first` once and a 200 afterwards, and with
 * setTimeout recording the retry delays it is asked for (then firing
 * immediately). The client's request-timeout timer (5000 ms) is left alone.
 */
async function captureRetryDelays(first: () => Response): Promise<{ delays: number[]; attempts: number }> {
  const originalFetch = globalThis.fetch;
  const originalSetTimeout = globalThis.setTimeout;
  const delays: number[] = [];
  let attempts = 0;
  globalThis.fetch = (async () => {
    attempts += 1;
    return attempts === 1 ? first() : okResponse();
  }) as typeof fetch;
  globalThis.setTimeout = ((fn: () => void, ms?: number) => {
    if (ms === 5000) return originalSetTimeout(fn, ms);
    delays.push(ms ?? 0);
    return originalSetTimeout(fn, 0);
  }) as typeof setTimeout;
  try {
    const client = new LiteLLMClient({ apiKey: "test-key", timeoutMs: 5000 });
    await client.complete("gpt-4o-mini", "system", "prompt", 0);
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.setTimeout = originalSetTimeout;
  }
  return { delays, attempts };
}

test("litellm client honours Retry-After seconds", async () => {
  const { delays, attempts } = await captureRetryDelays(
    () => new Response("slow down", { status: 429, headers: { "Retry-After": "7" } }),
  );
  assert.equal(attempts, 2);
  assert.deepEqual(delays, [7000]);
});

test("litellm client honours a Retry-After HTTP-date", async () => {
  const at = new Date(Date.now() + 20_000).toUTCString();
  const { delays } = await captureRetryDelays(
    () => new Response("slow down", { status: 503, headers: { "Retry-After": at } }),
  );
  assert.equal(delays.length, 1);
  // HTTP-dates have 1 s resolution.
  assert.ok(delays[0]! > 18_000 && delays[0]! <= 20_000, `delay ${delays[0]}`);
});

test("litellm client caps Retry-After at 60 s", async () => {
  const { delays } = await captureRetryDelays(
    () => new Response("slow down", { status: 429, headers: { "Retry-After": "3600" } }),
  );
  assert.deepEqual(delays, [60_000]);
});

test("litellm client uses exponential backoff without Retry-After", async () => {
  const { delays } = await captureRetryDelays(() => new Response("unavailable", { status: 503 }));
  assert.deepEqual(delays, [1000]);
});

test("litellm client maxAttempts bounds retries and is clamped to at least 1", async () => {
  const originalFetch = globalThis.fetch;
  let attempts = 0;
  globalThis.fetch = (async () => {
    attempts += 1;
    return new Response("unavailable", { status: 503, headers: { "Retry-After": "0" } });
  }) as typeof fetch;

  try {
    await assert.rejects(new LiteLLMClient({ apiKey: "k", maxAttempts: 2 }).complete("gpt-4o-mini", "s", "p", 0));
    assert.equal(attempts, 2);

    attempts = 0;
    await assert.rejects(new LiteLLMClient({ apiKey: "k", maxAttempts: 0 }).complete("gpt-4o-mini", "s", "p", 0));
    assert.equal(attempts, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("litellm client omits temperature for provider-prefixed reasoning models", async () => {
  const bodies: Array<Record<string, unknown>> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
    return okResponse();
  }) as typeof fetch;

  try {
    const client = new LiteLLMClient({ apiKey: "test-key" });
    await client.complete("openai/gpt-5-mini", "system", "prompt", 0.3);
    await client.complete("azure/o3-mini", "system", "prompt", 0.3);
    await client.complete("openai/gpt-4o-mini", "system", "prompt", 0.3);
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(Object.prototype.hasOwnProperty.call(bodies[0], "temperature"), false);
  assert.equal(Object.prototype.hasOwnProperty.call(bodies[1], "temperature"), false);
  assert.equal(bodies[2]!.temperature, 0.3);
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

// T7: timeout test — when fetch hangs past timeoutMs the AbortController must
// fire, fetch must reject with an AbortError, and withRetry must treat it as
// retryable so a slow first attempt can recover on the next one.
test("litellm client aborts a request that exceeds timeoutMs and retries", async () => {
  let attempts = 0;
  const originalFetch = globalThis.fetch;

  globalThis.fetch = ((_input: unknown, init?: RequestInit) => {
    attempts += 1;
    if (attempts === 1) {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }
    return Promise.resolve(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: "{\"label\":\"safe\",\"confidence\":0.9}" } }],
          usage: { total_tokens: 5 },
        }),
        { status: 200 },
      ),
    );
  }) as typeof fetch;

  try {
    const client = new LiteLLMClient({ apiKey: "test-key", timeoutMs: 20 });
    const result = await client.complete("gpt-4o-mini", "system", "prompt", 0);
    assert.equal(attempts, 2, "first attempt must be aborted; second must succeed");
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
