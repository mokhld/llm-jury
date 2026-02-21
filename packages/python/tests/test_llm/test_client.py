from __future__ import annotations

import sys
import types
import unittest

from llm_jury.llm.client import LiteLLMClient


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


if __name__ == "__main__":
    unittest.main()
