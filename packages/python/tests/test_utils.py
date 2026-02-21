from __future__ import annotations

import json

from llm_jury.utils import json_serializable, safe_json_parse, strip_markdown_fences


class TestStripMarkdownFences:
    def test_strip_markdown_fences_with_json_block(self) -> None:
        wrapped = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(wrapped) == '{"key": "value"}'

    def test_strip_markdown_fences_no_fences(self) -> None:
        plain = '{"key": "value"}'
        assert strip_markdown_fences(plain) == '{"key": "value"}'


class TestSafeJsonParse:
    def test_safe_json_parse_valid(self) -> None:
        result = safe_json_parse('{"a": 1, "b": 2}')
        assert result == {"a": 1, "b": 2}

    def test_safe_json_parse_invalid(self) -> None:
        assert safe_json_parse("not json at all") is None

    def test_safe_json_parse_non_dict(self) -> None:
        assert safe_json_parse("[1, 2, 3]") is None


class TestJsonSerializable:
    def test_json_serializable_set(self) -> None:
        result = json_serializable({3, 1, 2})
        assert result == [1, 2, 3]

    def test_json_serializable_numpy_like(self) -> None:
        class FakeArray:
            def tolist(self):
                return [1.0, 2.0, 3.0]

        result = json_serializable(FakeArray())
        assert result == [1.0, 2.0, 3.0]

    def test_json_serializable_fallback(self) -> None:
        class Custom:
            def __str__(self):
                return "custom-repr"

        result = json_serializable(Custom())
        assert result == "custom-repr"

    def test_json_serializable_roundtrip(self) -> None:
        """Verify json.dumps actually uses the handler without raising."""
        data = {"items": {3, 1, 2}, "name": "test"}
        output = json.dumps(data, default=json_serializable)
        parsed = json.loads(output)
        assert parsed["items"] == [1, 2, 3]
        assert parsed["name"] == "test"
