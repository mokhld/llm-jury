import test from "node:test";
import assert from "node:assert/strict";

import { buildPersonaResponseSchema } from "../../src/personas/schema.ts";

test("schema has OpenAI json_schema shape", () => {
  const rf = buildPersonaResponseSchema(["safe", "unsafe"]);
  assert.equal(rf.type, "json_schema");
  assert.equal(rf.json_schema.name, "persona_response");
  assert.equal(rf.json_schema.strict, true);

  const schema = rf.json_schema.schema as Record<string, unknown>;
  assert.equal(schema.type, "object");
  assert.equal(schema.additionalProperties, false);
  assert.deepEqual(
    [...(schema.required as string[])].sort(),
    ["confidence", "dissent_notes", "key_factors", "label", "reasoning"],
  );
});

test("label is constrained to provided labels via enum", () => {
  const rf = buildPersonaResponseSchema(["allow", "review", "reject"]);
  const props = (rf.json_schema.schema as { properties: Record<string, Record<string, unknown>> }).properties;
  assert.deepEqual(props.label!.enum, ["allow", "review", "reject"]);
});

test("empty labels omit the enum (no constraint)", () => {
  const rf = buildPersonaResponseSchema([]);
  const props = (rf.json_schema.schema as { properties: Record<string, Record<string, unknown>> }).properties;
  assert.equal("enum" in props.label!, false);
  assert.equal(props.label!.type, "string");
});

test("dissent_notes is nullable string", () => {
  const rf = buildPersonaResponseSchema(["a", "b"]);
  const props = (rf.json_schema.schema as { properties: Record<string, Record<string, unknown>> }).properties;
  assert.deepEqual(props.dissent_notes!.type, ["string", "null"]);
});
