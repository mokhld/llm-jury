from __future__ import annotations

import unittest

from llm_jury.classifiers.base import ClassificationResult
from llm_jury.debate.engine import DebateConfig, DebateEngine, DebateMode
from llm_jury.personas.base import Persona, PersonaResponse


class DebatePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.persona = Persona(
            name="Test Persona",
            role="Testing",
            system_prompt="You are testing",
        )
        self.primary = ClassificationResult(label="safe", confidence=0.51)

    def test_prompt_includes_input_and_labels(self) -> None:
        engine = DebateEngine(personas=[self.persona], config=DebateConfig())

        prompt = engine._build_persona_prompt(
            persona=self.persona,
            text="sample text",
            primary=self.primary,
            labels=["safe", "unsafe"],
            prior_rounds=[],
        )

        self.assertIn("sample text", prompt)
        self.assertIn("safe, unsafe", prompt)
        self.assertIn("Primary Classifier Result", prompt)

    def test_prompt_hides_primary_when_configured(self) -> None:
        engine = DebateEngine(
            personas=[self.persona],
            config=DebateConfig(include_primary_result=False),
        )

        prompt = engine._build_persona_prompt(
            persona=self.persona,
            text="sample text",
            primary=self.primary,
            labels=["safe", "unsafe"],
            prior_rounds=[],
        )

        self.assertNotIn("Primary Classifier Result", prompt)

    def test_prompt_includes_prior_rounds(self) -> None:
        engine = DebateEngine(personas=[self.persona], config=DebateConfig())
        prior = [[PersonaResponse("A", "safe", 0.8, "it is safe", ["context"])]]

        prompt = engine._build_persona_prompt(
            persona=self.persona,
            text="sample text",
            primary=self.primary,
            labels=["safe", "unsafe"],
            prior_rounds=prior,
        )

        self.assertIn("Round 1", prompt)
        self.assertIn("it is safe", prompt)

    def test_deliberation_prompt_includes_engagement_instructions(self) -> None:
        engine = DebateEngine(
            personas=[self.persona],
            config=DebateConfig(mode=DebateMode.DELIBERATION),
        )
        prior = [[PersonaResponse("A", "safe", 0.8, "it is safe", ["context"])]]

        prompt = engine._build_deliberation_prompt(
            persona=self.persona,
            text="sample text",
            primary=self.primary,
            labels=["safe", "unsafe"],
            prior_rounds=prior,
        )

        self.assertIn("Deliberation Instructions", prompt)
        self.assertIn("Engage with at least one other expert", prompt)

    def test_deliberation_prompt_labels_initial_opinions(self) -> None:
        engine = DebateEngine(
            personas=[self.persona],
            config=DebateConfig(mode=DebateMode.DELIBERATION),
        )
        prior = [[PersonaResponse("A", "safe", 0.8, "it is safe", ["context"])]]

        prompt = engine._build_deliberation_prompt(
            persona=self.persona,
            text="sample text",
            primary=self.primary,
            labels=["safe", "unsafe"],
            prior_rounds=prior,
        )

        self.assertIn("Initial Expert Opinions", prompt)
        self.assertNotIn("Previous Assessments", prompt)

    def test_adversarial_mode_assigns_role(self) -> None:
        engine = DebateEngine(
            personas=[self.persona],
            config=DebateConfig(mode=DebateMode.ADVERSARIAL),
        )

        prompt = engine._build_persona_prompt(
            persona=self.persona,
            text="sample text",
            primary=self.primary,
            labels=["safe", "unsafe"],
            prior_rounds=[],
        )

        self.assertIn("Adversarial Role", prompt)


if __name__ == "__main__":
    unittest.main()
