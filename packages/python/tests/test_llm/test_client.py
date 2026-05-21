from __future__ import annotations

import sys
import types
import unittest

from llm_jury.llm.client import LiteLLMClient, _is_retryable_error


def _fake_response(content: str = '{"label":"safe","confidence":0.9}'):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(total_tokens=5),
    )


class LiteLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_omits_temperature_for_o1(self) -> None:
        calls: list[dict[str, object]] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _fake_response()

        def fake_completion_cost(**kwargs):
            return 0.0

        original = sys.modules.get("litellm")
        sys.modules["litellm"] = types.SimpleNamespace(
            acompletion=fake_acompletion, completion_cost=fake_completion_cost,
        )
        try:
            client = LiteLLMClient()
            await client.complete("o1-preview", "system", "prompt", temperature=0.3)
        finally:
            if original is None:
                del sys.modules["litellm"]
            else:
                sys.modules["litellm"] = original

        self.assertEqual(len(calls), 1)
        self.assertNotIn("temperature", calls[0])

    async def test_omits_temperature_for_gpt5(self) -> None:
        calls: list[dict[str, object]] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _fake_response()

        def fake_completion_cost(**kwargs):
            return 0.0

        original = sys.modules.get("litellm")
        sys.modules["litellm"] = types.SimpleNamespace(
            acompletion=fake_acompletion, completion_cost=fake_completion_cost,
        )
        try:
            client = LiteLLMClient()
            await client.complete("gpt-5-mini", "system", "prompt", temperature=0.3)
        finally:
            if original is None:
                del sys.modules["litellm"]
            else:
                sys.modules["litellm"] = original

        self.assertEqual(len(calls), 1)
        self.assertNotIn("temperature", calls[0])

    async def test_includes_temperature_for_non_o1(self) -> None:
        calls: list[dict[str, object]] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _fake_response()

        def fake_completion_cost(**kwargs):
            return 0.0

        original = sys.modules.get("litellm")
        sys.modules["litellm"] = types.SimpleNamespace(
            acompletion=fake_acompletion, completion_cost=fake_completion_cost,
        )
        try:
            client = LiteLLMClient()
            await client.complete("gpt-4o-mini", "system", "prompt", temperature=0.3)
        finally:
            if original is None:
                del sys.modules["litellm"]
            else:
                sys.modules["litellm"] = original

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["temperature"], 0.3)


class RetryPredicateTests(unittest.TestCase):
    def test_connection_error_retries(self) -> None:
        self.assertTrue(_is_retryable_error(ConnectionError("boom")))

    def test_timeout_retries(self) -> None:
        self.assertTrue(_is_retryable_error(TimeoutError("slow")))

    def test_litellm_rate_limit_error_by_class_name_retries(self) -> None:
        # Synthesize a class that mimics litellm.RateLimitError without importing it.
        class RateLimitError(Exception):
            pass

        self.assertTrue(_is_retryable_error(RateLimitError("429")))

    def test_status_code_429_retries(self) -> None:
        exc = RuntimeError("rate limited")
        setattr(exc, "status_code", 429)
        self.assertTrue(_is_retryable_error(exc))

    def test_status_code_503_retries(self) -> None:
        exc = RuntimeError("unavailable")
        setattr(exc, "status_code", 503)
        self.assertTrue(_is_retryable_error(exc))

    def test_http_status_attribute_works(self) -> None:
        exc = RuntimeError("server error")
        setattr(exc, "http_status", 502)
        self.assertTrue(_is_retryable_error(exc))

    def test_status_attribute_works(self) -> None:
        exc = RuntimeError("server error")
        setattr(exc, "status", 500)
        self.assertTrue(_is_retryable_error(exc))

    def test_status_400_not_retried(self) -> None:
        exc = RuntimeError("bad request")
        setattr(exc, "status_code", 400)
        self.assertFalse(_is_retryable_error(exc))

    def test_generic_exception_not_retried(self) -> None:
        self.assertFalse(_is_retryable_error(ValueError("bad input")))


if __name__ == "__main__":
    unittest.main()
