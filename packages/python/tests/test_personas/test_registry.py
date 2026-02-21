from __future__ import annotations

import unittest

from llm_jury.personas.registry import PersonaRegistry


class PersonaRegistryTests(unittest.TestCase):
    def test_content_moderation_has_expected_personas(self) -> None:
        personas = PersonaRegistry.content_moderation()

        names = {persona.name for persona in personas}
        self.assertIn("Policy Analyst", names)
        self.assertIn("Cultural Context Expert", names)
        self.assertIn("Harm Assessment Specialist", names)
        self.assertTrue(all(persona.system_prompt for persona in personas))

    def test_custom_builder(self) -> None:
        personas = PersonaRegistry.custom(
            [
                {
                    "name": "Custom",
                    "role": "role",
                    "system_prompt": "prompt",
                    "model": "gpt-4o-mini",
                }
            ]
        )
        self.assertEqual(personas[0].name, "Custom")
        self.assertEqual(personas[0].model, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
