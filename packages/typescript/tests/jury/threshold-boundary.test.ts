import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { Jury } from "../../src/jury/core.ts";

test("confidence equal threshold does not escalate", async () => {
  const threshold = 0.7;
  const classifier = new FunctionClassifier(() => ["safe", threshold], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], confidenceThreshold: threshold });

  const verdict = await jury.classify("text");
  assert.equal(verdict.wasEscalated, false);
});
