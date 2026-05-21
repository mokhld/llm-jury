import test from "node:test";
import assert from "node:assert/strict";

import { CachingLLMClient } from "../../src/llm/cache.ts";
import type { LLMClient } from "../../src/llm/client.ts";

type Args = {
  model: string;
  systemPrompt: string;
  prompt: string;
  temperature: number | undefined;
  responseFormat: Record<string, unknown> | undefined;
};

function countingClient() {
  const calls: Args[] = [];
  const client: LLMClient = {
    async complete(model, systemPrompt, prompt, temperature, responseFormat) {
      calls.push({ model, systemPrompt, prompt, temperature, responseFormat });
      return {
        content: `${model}|${systemPrompt}|${prompt}|${temperature}|${JSON.stringify(responseFormat ?? null)}`,
        tokens: 10,
        costUsd: 0.001,
      };
    },
  };
  return { client, calls };
}

function failingClient() {
  let calls = 0;
  const client: LLMClient = {
    async complete() {
      calls++;
      throw new Error("boom");
    },
  };
  return {
    client,
    get calls() {
      return calls;
    },
  };
}

test("repeated call hits cache and skips inner", async () => {
  const { client, calls } = countingClient();
  const cache = new CachingLLMClient(client);

  const a = await cache.complete("m", "sys", "p", 0, undefined);
  const b = await cache.complete("m", "sys", "p", 0, undefined);

  assert.equal(calls.length, 1);
  assert.equal(cache.hits, 1);
  assert.equal(cache.misses, 1);
  assert.deepEqual(a, b);
});

test("different keys are distinct", async () => {
  const { client, calls } = countingClient();
  const cache = new CachingLLMClient(client);

  await cache.complete("m1", "sys", "p", 0, undefined);
  await cache.complete("m2", "sys", "p", 0, undefined);
  await cache.complete("m1", "sys2", "p", 0, undefined);
  await cache.complete("m1", "sys", "p2", 0, undefined);
  await cache.complete("m1", "sys", "p", 0.5, undefined);
  await cache.complete("m1", "sys", "p", 0, { x: 1 });

  assert.equal(calls.length, 6);
  assert.equal(cache.hits, 0);
  assert.equal(cache.misses, 6);
});

test("response_format key is order-stable", async () => {
  const { client, calls } = countingClient();
  const cache = new CachingLLMClient(client);

  await cache.complete("m", "sys", "p", 0, { a: 1, b: 2 });
  await cache.complete("m", "sys", "p", 0, { b: 2, a: 1 });

  assert.equal(calls.length, 1);
  assert.equal(cache.hits, 1);
});

test("errors are not cached", async () => {
  const failing = failingClient();
  const cache = new CachingLLMClient(failing.client);

  await assert.rejects(cache.complete("m", "sys", "p", 0, undefined), /boom/);
  await assert.rejects(cache.complete("m", "sys", "p", 0, undefined), /boom/);

  assert.equal(failing.calls, 2);
  assert.equal(cache.size, 0);
  assert.equal(cache.misses, 0);
  assert.equal(cache.hits, 0);
});

test("LRU eviction drops oldest entry", async () => {
  const { client, calls } = countingClient();
  const cache = new CachingLLMClient(client, { maxSize: 2 });

  await cache.complete("m", "sys", "p1", 0, undefined);
  await cache.complete("m", "sys", "p2", 0, undefined);
  await cache.complete("m", "sys", "p3", 0, undefined); // evicts p1

  assert.equal(cache.size, 2);

  await cache.complete("m", "sys", "p1", 0, undefined); // miss — was evicted
  assert.equal(calls.length, 4);
});

test("access promotes entry in LRU order", async () => {
  const { client } = countingClient();
  const cache = new CachingLLMClient(client, { maxSize: 2 });

  await cache.complete("m", "sys", "p1", 0, undefined); // A
  await cache.complete("m", "sys", "p2", 0, undefined); // B
  await cache.complete("m", "sys", "p1", 0, undefined); // hits A, promotes
  await cache.complete("m", "sys", "p3", 0, undefined); // evicts B, not A

  await cache.complete("m", "sys", "p1", 0, undefined); // still cached → hit
  assert.equal(cache.hits, 2);
});

test("TTL expires old entries", async () => {
  const { client, calls } = countingClient();
  const cache = new CachingLLMClient(client, { ttlSeconds: 0.05 });

  await cache.complete("m", "sys", "p", 0, undefined);
  await new Promise((r) => setTimeout(r, 100));
  await cache.complete("m", "sys", "p", 0, undefined);

  assert.equal(calls.length, 2);
  assert.equal(cache.misses, 2);
});

test("clear resets state", async () => {
  const { client, calls } = countingClient();
  const cache = new CachingLLMClient(client);

  await cache.complete("m", "sys", "p", 0, undefined);
  await cache.complete("m", "sys", "p", 0, undefined);
  assert.equal(cache.hits, 1);

  cache.clear();
  assert.equal(cache.size, 0);
  assert.equal(cache.hits, 0);
  assert.equal(cache.misses, 0);

  await cache.complete("m", "sys", "p", 0, undefined);
  assert.equal(calls.length, 2);
});

test("invalid options rejected", () => {
  const { client } = countingClient();
  assert.throws(() => new CachingLLMClient(client, { maxSize: 0 }), /maxSize/);
  assert.throws(() => new CachingLLMClient(client, { maxSize: -1 }), /maxSize/);
  assert.throws(() => new CachingLLMClient(client, { ttlSeconds: 0 }), /ttlSeconds/);
  assert.throws(() => new CachingLLMClient(client, { ttlSeconds: -1 }), /ttlSeconds/);
});
