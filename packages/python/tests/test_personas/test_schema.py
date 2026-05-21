from __future__ import annotations

import unittest

from llm_jury.personas.schema import build_persona_response_schema


class PersonaResponseSchemaTests(unittest.TestCase):
    def test_shape_matches_openai_json_schema_format(self) -> None:
        rf = build_persona_response_schema(["safe", "unsafe"])

        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["name"], "persona_response")
        self.assertIs(rf["json_schema"]["strict"], True)

        schema = rf["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(
            sorted(schema["required"]),
            sorted(
                ["label", "confidence", "reasoning", "key_factors", "dissent_notes"]
            ),
        )

    def test_label_constrained_to_provided_labels(self) -> None:
        rf = build_persona_response_schema(["allow", "review", "reject"])
        label_schema = rf["json_schema"]["schema"]["properties"]["label"]
        self.assertEqual(label_schema["enum"], ["allow", "review", "reject"])

    def test_empty_labels_omit_enum(self) -> None:
        rf = build_persona_response_schema([])
        label_schema = rf["json_schema"]["schema"]["properties"]["label"]
        self.assertNotIn("enum", label_schema)
        self.assertEqual(label_schema["type"], "string")

    def test_dissent_notes_is_nullable_string(self) -> None:
        rf = build_persona_response_schema(["a", "b"])
        dn = rf["json_schema"]["schema"]["properties"]["dissent_notes"]
        self.assertEqual(dn["type"], ["string", "null"])


if __name__ == "__main__":
    unittest.main()
