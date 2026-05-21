/**
 * JSON Schema for persona responses.
 *
 * The schema is sent to the LLM as OpenAI-style `response_format` so the
 * provider itself enforces well-formed output. LiteLLM-compatible endpoints
 * forward the schema to any provider that supports structured output;
 * non-supporting providers fall back to prompt-only enforcement.
 *
 * Strict-mode JSON Schema requires `additionalProperties: false` and every
 * property listed in `required` — optional fields like `dissent_notes` are
 * modeled as nullable strings instead of being omitted.
 */

export type PersonaResponseFormat = {
  type: "json_schema";
  json_schema: {
    name: string;
    schema: Record<string, unknown>;
    strict: true;
  };
};

export function buildPersonaResponseSchema(labels: string[]): PersonaResponseFormat {
  const labelProperty: Record<string, unknown> = { type: "string" };
  if (labels.length > 0) {
    labelProperty.enum = [...labels];
  }

  const schema: Record<string, unknown> = {
    type: "object",
    properties: {
      label: labelProperty,
      confidence: { type: "number" },
      reasoning: { type: "string" },
      key_factors: { type: "array", items: { type: "string" } },
      dissent_notes: { type: ["string", "null"] },
    },
    required: ["label", "confidence", "reasoning", "key_factors", "dissent_notes"],
    additionalProperties: false,
  };

  return {
    type: "json_schema",
    json_schema: {
      name: "persona_response",
      schema,
      strict: true,
    },
  };
}
