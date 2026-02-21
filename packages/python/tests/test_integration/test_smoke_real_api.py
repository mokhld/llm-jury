from __future__ import annotations

import os
import unittest


def _has_real_api_prereqs() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import litellm  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_real_api_prereqs(), "Requires OPENAI_API_KEY and litellm installed")
class RealAPISmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_smoke_classification_pipeline(self) -> None:
        from llm_jury.classifiers.llm_classifier import LLMClassifier
        from llm_jury.judges.majority_vote import MajorityVoteJudge
        from llm_jury.jury.core import Jury
        from llm_jury.personas.base import Persona

        classifier = LLMClassifier(labels=["safe", "unsafe"])

        personas = [
            Persona(
                name="Policy Analyst",
                role="Policy reasoning",
                system_prompt="Assess policy risk conservatively.",
            ),
            Persona(
                name="Context Expert",
                role="Context reasoning",
                system_prompt="Assess context and non-literal usage.",
            ),
        ]

        jury = Jury(
            classifier=classifier,
            personas=personas,
            confidence_threshold=1.01,
            judge=MajorityVoteJudge(),
            max_debate_cost_usd=2.0,
        )

        verdict = await jury.classify("Hello, how are you?")
        self.assertIn(verdict.label, {"safe", "unsafe"})
        self.assertGreaterEqual(verdict.confidence, 0.0)
        self.assertLessEqual(verdict.confidence, 1.0)
        self.assertTrue(verdict.was_escalated)


if __name__ == "__main__":
    unittest.main()
