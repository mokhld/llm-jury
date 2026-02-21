import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { Jury } from "../../src/jury/core.ts";
import { ThresholdCalibrator } from "../../src/calibration/optimizer.ts";

test("threshold calibrator returns candidate threshold", async () => {
  const values: Record<string, [string, number]> = {
    a: ["safe", 0.9],
    b: ["unsafe", 0.45],
    c: ["safe", 0.65],
    d: ["unsafe", 0.55],
  };

  const classifier = new FunctionClassifier((text: string) => values[text], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });
  const calibrator = new ThresholdCalibrator(jury);

  const threshold = await calibrator.calibrate({
    texts: ["a", "b", "c", "d"],
    labels: ["safe", "unsafe", "safe", "unsafe"],
    thresholds: [0.5, 0.6, 0.7, 0.8],
    errorCost: 10,
    escalationCost: 0.05,
  });

  assert.ok([0.5, 0.6, 0.7, 0.8].includes(threshold));
  assert.equal(calibrator.calibrationReport().bestThreshold, threshold);
});
