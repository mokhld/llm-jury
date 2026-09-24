import test from "node:test";
import assert from "node:assert/strict";

import { HuggingFaceClassifier } from "../../src/classifiers/huggingFaceAdapter.ts";
import type { HuggingFaceLabelScore } from "../../src/classifiers/huggingFaceAdapter.ts";
import { SklearnClassifier } from "../../src/classifiers/sklearnAdapter.ts";

const FULL_SCORES: HuggingFaceLabelScore[] = [
  { label: "unsafe", score: 0.7 },
  { label: "safe", score: 0.2 },
  { label: "spam", score: 0.1 },
];

// --- HuggingFaceClassifier -------------------------------------------------------

test("huggingface adapter asks the pipeline for every label's score", async () => {
  const calls: Array<Record<string, unknown> | undefined> = [];
  const classifier = new HuggingFaceClassifier({
    pipeline: async (_text, options) => {
      calls.push(options);
      return FULL_SCORES;
    },
  });
  await classifier.classify("text");
  // `topk` for @xenova/transformers v2, `top_k` for @huggingface/transformers v3.
  assert.deepEqual(calls, [{ topk: null, top_k: null }]);
});

test("huggingface adapter derives labels from a full score list", async () => {
  const classifier = new HuggingFaceClassifier({ pipeline: async () => [FULL_SCORES] });
  const result = await classifier.classify("text");
  assert.equal(result.label, "unsafe");
  assert.equal(result.confidence, 0.7);
  assert.deepEqual(result.rawOutput, FULL_SCORES);
  assert.deepEqual(classifier.labels, ["unsafe", "safe", "spam"]);
});

test("huggingface adapter does not lock labels to a top-1 result", async () => {
  let full = false;
  const classifier = new HuggingFaceClassifier({
    pipeline: async () => (full ? FULL_SCORES : [{ label: "unsafe", score: 0.7 }]),
  });
  await classifier.classify("first");
  assert.deepEqual(classifier.labels, []);
  full = true;
  await classifier.classify("second");
  assert.deepEqual(classifier.labels, ["unsafe", "safe", "spam"]);
});

test("huggingface adapter keeps explicit labels", async () => {
  const classifier = new HuggingFaceClassifier({ labels: ["safe", "unsafe"], pipeline: async () => FULL_SCORES });
  await classifier.classify("text");
  assert.deepEqual(classifier.labels, ["safe", "unsafe"]);
});

test("huggingface adapter reports zero cost and rejects empty output", async () => {
  const classifier = new HuggingFaceClassifier({ pipeline: async () => FULL_SCORES });
  assert.equal((await classifier.classify("text")).costUsd, 0);

  const empty = new HuggingFaceClassifier({ pipeline: async () => [] });
  await assert.rejects(empty.classify("text"), /returned no scores/);
});

// --- SklearnClassifier -----------------------------------------------------------

test("sklearn adapter reports zero cost", async () => {
  const classifier = new SklearnClassifier({ predictProba: () => [[0.25, 0.75]] }, ["a", "b"]);
  const result = await classifier.classify("text");
  assert.equal(result.label, "b");
  assert.equal(result.costUsd, 0);
});

test("sklearn adapter maps columns by model.classes when they match the labels", async () => {
  // scikit-learn sorts classes, so the columns are [safe, unsafe] even though
  // the labels were listed the other way round.
  const model = { classes: ["safe", "unsafe"], predictProba: () => [[0.9, 0.1]] };
  const classifier = new SklearnClassifier(model, ["unsafe", "safe"]);
  const result = await classifier.classify("text");
  assert.equal(result.label, "safe");
  assert.equal(result.confidence, 0.9);
  assert.deepEqual(classifier.labels, ["unsafe", "safe"]);
});

test("sklearn adapter maps labels by position for non-string classes", async () => {
  const model = { classes: [0, 1], predictProba: () => [[0.2, 0.8]] };
  const classifier = new SklearnClassifier(model, ["ham", "spam"]);
  assert.equal((await classifier.classify("text")).label, "spam");
});

test("sklearn adapter rejects a label count that differs from model.classes", () => {
  const model = { classes: ["a", "b", "c"], predictProba: () => [[0.2, 0.3, 0.5]] };
  assert.throws(() => new SklearnClassifier(model, ["a", "b"]), /3 entries .* 2 labels/);
});
