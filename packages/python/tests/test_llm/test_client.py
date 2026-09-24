from __future__ import annotations

import contextlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest import mock

from llm_jury.llm import client as client_module
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
            acompletion=fake_acompletion,
            completion_cost=fake_completion_cost,
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
            acompletion=fake_acompletion,
            completion_cost=fake_completion_cost,
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
            acompletion=fake_acompletion,
            completion_cost=fake_completion_cost,
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


@contextlib.contextmanager
def _fake_litellm(acompletion):
    def fake_completion_cost(**kwargs):
        return 0.0

    original = sys.modules.get("litellm")
    sys.modules["litellm"] = types.SimpleNamespace(
        acompletion=acompletion,
        completion_cost=fake_completion_cost,
    )
    try:
        yield
    finally:
        if original is None:
            del sys.modules["litellm"]
        else:
            sys.modules["litellm"] = original


class RateLimitError(Exception):
    """Mimics litellm.RateLimitError (matched by class name)."""

    def __init__(self, message: str = "429", headers=None, response_headers=None):
        super().__init__(message)
        if headers is not None:
            self.headers = headers
        if response_headers is not None:
            self.response = types.SimpleNamespace(headers=response_headers)


class LiteLLMClientTransportTests(unittest.IsolatedAsyncioTestCase):
    """BUG-08: timeouts, credentials, configurable retries and Retry-After."""

    def setUp(self) -> None:
        self.sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        patcher = mock.patch.object(client_module, "_sleep", fake_sleep)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _call(self, client: LiteLLMClient, acompletion, model="gpt-4o-mini"):
        with _fake_litellm(acompletion):
            return await client.complete(model, "system", "prompt", temperature=0.0)

    async def test_default_timeout_is_sixty_seconds(self) -> None:
        calls: list[dict] = []

        async def acompletion(**kwargs):
            calls.append(kwargs)
            return _fake_response()

        await self._call(LiteLLMClient(), acompletion)

        self.assertEqual(calls[0]["timeout"], 60.0)
        self.assertNotIn("api_key", calls[0])
        self.assertNotIn("api_base", calls[0])

    async def test_timeout_and_credentials_are_forwarded(self) -> None:
        calls: list[dict] = []

        async def acompletion(**kwargs):
            calls.append(kwargs)
            return _fake_response()

        client = LiteLLMClient(
            timeout_seconds=5.0, api_key="sk-test", api_base="http://proxy:4000"
        )
        await self._call(client, acompletion)

        self.assertEqual(calls[0]["timeout"], 5.0)
        self.assertEqual(calls[0]["api_key"], "sk-test")
        self.assertEqual(calls[0]["api_base"], "http://proxy:4000")

    async def test_timeout_none_is_not_sent(self) -> None:
        calls: list[dict] = []

        async def acompletion(**kwargs):
            calls.append(kwargs)
            return _fake_response()

        await self._call(LiteLLMClient(timeout_seconds=None), acompletion)

        self.assertNotIn("timeout", calls[0])

    async def test_max_attempts_is_configurable(self) -> None:
        attempts = 0

        async def acompletion(**kwargs):
            nonlocal attempts
            attempts += 1
            raise RateLimitError()

        with self.assertRaises(RateLimitError):
            await self._call(LiteLLMClient(max_attempts=5), acompletion)
        self.assertEqual(attempts, 5)

        attempts = 0
        with self.assertRaises(RateLimitError):
            await self._call(LiteLLMClient(max_attempts=1), acompletion)
        self.assertEqual(attempts, 1)

    async def test_retries_then_succeeds_with_exponential_backoff(self) -> None:
        attempts = 0

        async def acompletion(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("reset")
            return _fake_response()

        result = await self._call(LiteLLMClient(), acompletion)

        self.assertEqual(attempts, 3)
        self.assertEqual(result["tokens"], 5)
        self.assertEqual(self.sleeps, [1.0, 2.0])

    async def test_non_retryable_error_is_not_retried(self) -> None:
        attempts = 0

        async def acompletion(**kwargs):
            nonlocal attempts
            attempts += 1
            raise ValueError("bad request")

        with self.assertRaises(ValueError):
            await self._call(LiteLLMClient(), acompletion)
        self.assertEqual(attempts, 1)

    async def _sleep_for(self, exc: BaseException) -> float:
        raised = False

        async def acompletion(**kwargs):
            nonlocal raised
            if not raised:
                raised = True
                raise exc
            return _fake_response()

        await self._call(LiteLLMClient(), acompletion)
        self.assertEqual(len(self.sleeps), 1)
        return self.sleeps.pop()

    async def test_retry_after_seconds_on_response_headers(self) -> None:
        delay = await self._sleep_for(
            RateLimitError(response_headers={"retry-after": "7"})
        )
        self.assertEqual(delay, 7.0)

    async def test_retry_after_on_exception_headers(self) -> None:
        delay = await self._sleep_for(RateLimitError(headers={"Retry-After": "3"}))
        self.assertEqual(delay, 3.0)

    async def test_retry_after_is_capped_at_sixty_seconds(self) -> None:
        delay = await self._sleep_for(
            RateLimitError(response_headers={"retry-after": "3600"})
        )
        self.assertEqual(delay, 60.0)

    async def test_retry_after_http_date(self) -> None:
        when = datetime.now(timezone.utc) + timedelta(seconds=30)
        delay = await self._sleep_for(
            RateLimitError(headers={"retry-after": format_datetime(when, usegmt=True)})
        )
        self.assertGreater(delay, 20.0)
        self.assertLessEqual(delay, 30.0)

    async def test_unparseable_retry_after_uses_backoff(self) -> None:
        delay = await self._sleep_for(RateLimitError(headers={"retry-after": "soon"}))
        self.assertEqual(delay, 1.0)

    async def _sends_temperature(self, model: str) -> bool:
        calls: list[dict] = []

        async def acompletion(**kwargs):
            calls.append(kwargs)
            return _fake_response()

        with _fake_litellm(acompletion):
            await LiteLLMClient().complete(model, "s", "p", temperature=0.3)
        return "temperature" in calls[0]

    async def test_provider_prefixed_reasoning_models_omit_temperature(self) -> None:
        self.assertFalse(await self._sends_temperature("openai/gpt-5-mini"))
        self.assertFalse(await self._sends_temperature("azure/o3-mini"))
        self.assertFalse(await self._sends_temperature("openrouter/openai/o1-preview"))
        self.assertTrue(await self._sends_temperature("anthropic/claude-sonnet-4"))


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
        exc.status_code = 429
        self.assertTrue(_is_retryable_error(exc))

    def test_status_code_503_retries(self) -> None:
        exc = RuntimeError("unavailable")
        exc.status_code = 503
        self.assertTrue(_is_retryable_error(exc))

    def test_http_status_attribute_works(self) -> None:
        exc = RuntimeError("server error")
        exc.http_status = 502
        self.assertTrue(_is_retryable_error(exc))

    def test_status_attribute_works(self) -> None:
        exc = RuntimeError("server error")
        exc.status = 500
        self.assertTrue(_is_retryable_error(exc))

    def test_status_400_not_retried(self) -> None:
        exc = RuntimeError("bad request")
        exc.status_code = 400
        self.assertFalse(_is_retryable_error(exc))

    def test_generic_exception_not_retried(self) -> None:
        self.assertFalse(_is_retryable_error(ValueError("bad input")))


if __name__ == "__main__":
    unittest.main()
