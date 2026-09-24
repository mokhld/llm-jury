/**
 * JSON Schemas for structured LLM output (persona, classifier and judge).
 *
 * Each schema is sent to the LLM as OpenAI-style `response_format` so the
 * provider itself enforces well-formed output. LiteLLM-compatible endpoints
 * forward the schema to any provider that supports structured output;
 * non-supporting providers fall back to prompt-only enforcement, which is why
 * callers still validate the parsed label and confidence.
 *
 * Strict-mode JSON Schema requires `additionalProperties: false` and every
 * property listed in `required`. Optional fields like `dissent_notes` are
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

function labelProperty(labels: string[]): Record<string, unknown> {
  const property: Record<string, unknown> = { type: "string" };
  if (labels.length > 0) {
    property.enum = [...labels];
  }
  return property;
}

function strictResponseFormat(
  name: string,
  properties: Record<string, unknown>,
): PersonaResponseFormat {
  return {
    type: "json_schema",
    json_schema: {
      name,
      schema: {
        type: "object",
        properties,
        required: Object.keys(properties),
        additionalProperties: false,
      },
      strict: true,
    },
  };
}

export function buildPersonaResponseSchema(labels: string[]): PersonaResponseFormat {
  return strictResponseFormat("persona_response", {
    label: labelProperty(labels),
    confidence: { type: "number" },
    reasoning: { type: "string" },
    key_factors: { type: "array", items: { type: "string" } },
    dissent_notes: { type: ["string", "null"] },
  });
}

/** Response format for `LLMClassifier`: a label from `labels` and a confidence. */
export function buildClassifierResponseSchema(labels: string[]): PersonaResponseFormat {
  return strictResponseFormat("classifier_response", {
    label: labelProperty(labels),
    confidence: { type: "number" },
  });
}

/** Response format for `LLMJudge`: the verdict plus its audit fields. */
export function buildJudgeResponseSchema(labels: string[]): PersonaResponseFormat {
  return strictResponseFormat("judge_response", {
    label: labelProperty(labels),
    confidence: { type: "number" },
    reasoning: { type: "string" },
    key_agreements: { type: "array", items: { type: "string" } },
    key_disagreements: { type: "array", items: { type: "string" } },
    decisive_factor: { type: "string" },
  });
}
