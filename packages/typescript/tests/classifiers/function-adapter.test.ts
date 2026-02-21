import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";

test("FunctionClassifier classify", async () => {
  const classifier = new FunctionClassifier((text: string) => ["safe", 0.9], ["safe", "unsafe"]);
  const result = await classifier.classify("hello");

  assert.equal(result.label, "safe");
  assert.equal(result.confidence, 0.9);
});

test("FunctionClassifier classifyBatch", async () => {
  const classifier = new FunctionClassifier(
    (text: string) => (text.includes("!") ? ["unsafe", 0.6] : ["safe", 0.95]),
    ["safe", "unsafe"],
  );
  const results = await classifier.classifyBatch(["a", "b!"]);

  assert.deepEqual(results.map((x) => x.label), ["safe", "unsafe"]);
});
