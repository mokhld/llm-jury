from __future__ import annotations

import json
import re
import unittest

from llm_jury._version import __version__
from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateTranscript
from llm_jury.judges.base import Verdict
from llm_jury.judges.majority_vote import MajorityVoteJudge
from llm_jury.personas.base import PersonaResponse

_ISO_8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _transcript() -> DebateTranscript:
    return DebateTranscript(
        input_text="text",
        primary_result=ClassificationResult("unknown", 0.4),
        rounds=[
            [
                PersonaResponse("A", "unsafe", 0.9, "r1", ["a"]),
                PersonaResponse("B", "unsafe", 0.8, "r2", ["b"]),
                PersonaResponse("C", "unsafe", 0.7, "r3", ["c"]),
            ]
        ],
        duration_ms=10,
        total_tokens=20,
        total_cost_usd=0.001,
    )


class VerdictProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_verdict_includes_library_version(self) -> None:
        verdict = await MajorityVoteJudge().judge(_transcript(), ["safe", "unsafe"])
        self.assertEqual(verdict.library_version, __version__)

    async def test_verdict_includes_created_at_iso8601(self) -> None:
        verdict = await MajorityVoteJudge().judge(_transcript(), ["safe", "unsafe"])
        self.assertIsNotNone(_ISO_8601_RE.match(verdict.created_at))

    async def test_verdict_to_dict_round_trips_provenance(self) -> None:
        verdict = await MajorityVoteJudge().judge(_transcript(), ["safe", "unsafe"])
        data = verdict.to_dict()
        self.assertIn("library_version", data)
        self.assertIn("created_at", data)
        self.assertEqual(data["library_version"], __version__)

    async def test_verdict_to_json_emits_provenance(self) -> None:
        verdict = await MajorityVoteJudge().judge(_transcript(), ["safe", "unsafe"])
        payload = json.loads(verdict.to_json())
        self.assertEqual(payload["library_version"], __version__)
        self.assertTrue(_ISO_8601_RE.match(payload["created_at"]))

    def test_verdict_explicit_init_overrides_defaults(self) -> None:
        verdict = Verdict(
            label="x",
            confidence=0.5,
            reasoning="r",
            was_escalated=False,
            primary_result=ClassificationResult("x", 0.5),
            debate_transcript=None,
            judge_strategy="test",
            total_duration_ms=1,
            total_cost_usd=None,
            library_version="9.9.9",
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(verdict.library_version, "9.9.9")
        self.assertEqual(verdict.created_at, "2026-01-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
