import test from "node:test";
import assert from "node:assert/strict";

import { HuggingFaceClassifier } from "../../src/classifiers/huggingFaceAdapter.ts";
import { SklearnClassifier } from "../../src/classifiers/sklearnAdapter.ts";
import { LLMClassifier } from "../../src/classifiers/llmClassifier.ts";
import { FakeLLMClient } from "../helpers.ts";

test("sklearn adapter uses predictProba", async () => {
  const model = {
    predictProba: async (_features: unknown) => [[0.1, 0.9]],
  };
  const classifier = new SklearnClassifier(model, ["safe", "unsafe"]);
  const result = await classifier.classify("text");
  assert.equal(result.label, "unsafe");
  assert.equal(result.confidence, 0.9);
});

test("huggingface adapter uses injected pipeline", async () => {
  const classifier = new HuggingFaceClassifier({
    pipeline: async (_text: string) => [
      { label: "safe", score: 0.2 },
      { label: "unsafe", score: 0.8 },
    ],
  });
  const result = await classifier.classify("text");
  assert.equal(result.label, "unsafe");
  assert.equal(result.confidence, 0.8);
});

test("llm classifier parses JSON", async () => {
  const llm = new FakeLLMClient({
    classifier: { content: JSON.stringify({ label: "unsafe", confidence: 0.82 }) },
  });
  const classifier = new LLMClassifier({
    model: "classifier",
    labels: ["safe", "unsafe"],
    llmClient: llm,
  });
  const result = await classifier.classify("text");
  assert.equal(result.label, "unsafe");
  assert.equal(result.confidence, 0.82);
});

test("llm classifier falls back on invalid JSON", async () => {
  const llm = new FakeLLMClient({
    classifier: { content: "not-json" },
  });
  const classifier = new LLMClassifier({
    model: "classifier",
    labels: ["safe", "unsafe"],
    llmClient: llm,
  });
  const result = await classifier.classify("text");
  assert.equal(result.label, "safe");
  assert.equal(result.confidence, 0);
});
