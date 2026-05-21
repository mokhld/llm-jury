from __future__ import annotations

import asyncio
import unittest

from llm_jury.llm.cache import CachingLLMClient


class _CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float | None = 0.0,
        response_format: dict | None = None,
    ) -> dict:
        self.calls += 1
        return {
            "content": f"{model}|{system_prompt}|{prompt}|{temperature}|{response_format}",
            "tokens": 10,
            "cost_usd": 0.001,
        }


class _FailingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, model, system_prompt, prompt, temperature=0.0, response_format=None
    ):
        self.calls += 1
        raise RuntimeError("boom")


class CachingLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_call_hits_cache_and_skips_inner(self) -> None:
        inner = _CountingClient()
        cache = CachingLLMClient(inner)

        a = await cache.complete("m", "sys", "p", 0.0, None)
        b = await cache.complete("m", "sys", "p", 0.0, None)

        self.assertEqual(a, b)
        self.assertEqual(inner.calls, 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    async def test_different_keys_are_distinct(self) -> None:
        inner = _CountingClient()
        cache = CachingLLMClient(inner)

        await cache.complete("m1", "sys", "p", 0.0, None)
        await cache.complete("m2", "sys", "p", 0.0, None)  # model differs
        await cache.complete("m1", "sys2", "p", 0.0, None)  # system_prompt differs
        await cache.complete("m1", "sys", "p2", 0.0, None)  # prompt differs
        await cache.complete("m1", "sys", "p", 0.5, None)  # temperature differs
        await cache.complete("m1", "sys", "p", 0.0, {"x": 1})  # response_format differs

        self.assertEqual(inner.calls, 6)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 6)

    async def test_response_format_key_is_order_stable(self) -> None:
        # Two semantically-identical response_format dicts with different
        # insertion order must produce the same cache key.
        inner = _CountingClient()
        cache = CachingLLMClient(inner)

        await cache.complete("m", "sys", "p", 0.0, {"a": 1, "b": 2})
        await cache.complete("m", "sys", "p", 0.0, {"b": 2, "a": 1})

        self.assertEqual(inner.calls, 1)
        self.assertEqual(cache.hits, 1)

    async def test_errors_are_not_cached(self) -> None:
        inner = _FailingClient()
        cache = CachingLLMClient(inner)

        with self.assertRaises(RuntimeError):
            await cache.complete("m", "sys", "p", 0.0, None)
        with self.assertRaises(RuntimeError):
            await cache.complete("m", "sys", "p", 0.0, None)

        # Both calls hit the inner client; nothing was cached.
        self.assertEqual(inner.calls, 2)
        self.assertEqual(cache.misses, 0)
        self.assertEqual(cache.hits, 0)

    async def test_lru_eviction_drops_oldest_entry(self) -> None:
        inner = _CountingClient()
        cache = CachingLLMClient(inner, max_size=2)

        await cache.complete("m", "sys", "p1", 0.0, None)  # entry A
        await cache.complete("m", "sys", "p2", 0.0, None)  # entry B
        await cache.complete("m", "sys", "p3", 0.0, None)  # evicts A

        self.assertEqual(cache.size, 2)

        # Re-requesting A should miss (it was evicted)
        await cache.complete("m", "sys", "p1", 0.0, None)
        self.assertEqual(inner.calls, 4)

    async def test_access_promotes_entry_in_lru_order(self) -> None:
        inner = _CountingClient()
        cache = CachingLLMClient(inner, max_size=2)

        await cache.complete("m", "sys", "p1", 0.0, None)  # entry A
        await cache.complete("m", "sys", "p2", 0.0, None)  # entry B
        await cache.complete("m", "sys", "p1", 0.0, None)  # hits A, promotes it
        await cache.complete("m", "sys", "p3", 0.0, None)  # evicts B (LRU), not A

        # A should still be cached:
        await cache.complete("m", "sys", "p1", 0.0, None)
        self.assertEqual(cache.hits, 2)  # one hit before eviction, one after

    async def test_ttl_expires_old_entries(self) -> None:
        inner = _CountingClient()
        cache = CachingLLMClient(inner, ttl_seconds=0.05)

        await cache.complete("m", "sys", "p", 0.0, None)
        await asyncio.sleep(0.1)
        await cache.complete("m", "sys", "p", 0.0, None)

        self.assertEqual(inner.calls, 2)
        self.assertEqual(cache.misses, 2)

    async def test_clear_resets_state(self) -> None:
        inner = _CountingClient()
        cache = CachingLLMClient(inner)

        await cache.complete("m", "sys", "p", 0.0, None)
        await cache.complete("m", "sys", "p", 0.0, None)
        self.assertEqual(cache.hits, 1)

        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

        await cache.complete("m", "sys", "p", 0.0, None)
        self.assertEqual(inner.calls, 2)

    def test_invalid_options_rejected(self) -> None:
        inner = _CountingClient()
        with self.assertRaises(ValueError):
            CachingLLMClient(inner, max_size=0)
        with self.assertRaises(ValueError):
            CachingLLMClient(inner, max_size=-1)
        with self.assertRaises(ValueError):
            CachingLLMClient(inner, ttl_seconds=0)
        with self.assertRaises(ValueError):
            CachingLLMClient(inner, ttl_seconds=-1)
