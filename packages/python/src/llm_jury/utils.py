from __future__ import annotations

import json


def strip_markdown_fences(content: str) -> str:
    """Remove markdown code fences (```json ... ```) wrapping a JSON payload."""
    text = content.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


def safe_json_parse(content: str) -> dict | None:
    """Parse a JSON string, returning *None* on failure instead of raising."""
    try:
        data = json.loads(content)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def clamp_confidence(value: float) -> float:
    """Clamp a confidence value to the valid [0.0, 1.0] range."""
    return max(0.0, min(1.0, float(value)))


def json_serializable(obj: object) -> object:
    """Default handler for :func:`json.dumps` that gracefully converts
    non-serializable types (numpy arrays, sets, etc.) to JSON-safe primitives."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)
