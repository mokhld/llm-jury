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

// T8: empty input — must not throw and must produce a defined report row per threshold.
test("threshold calibrator handles empty input without crashing", async () => {
  const classifier = new FunctionClassifier(() => ["safe", 0.9], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });
  const calibrator = new ThresholdCalibrator(jury);

  const threshold = await calibrator.calibrate({
    texts: [],
    labels: [],
    thresholds: [0.5, 0.7, 0.9],
  });

  // With zero samples every threshold has totalCost=0; first one wins.
  assert.equal(threshold, 0.5);
  const report = calibrator.calibrationReport();
  assert.equal(report.rows.length, 3);
  for (const row of report.rows) {
    assert.equal(row.totalCost, 0);
    assert.equal(row.escalationRate, 0);
    // accuracy is correct/max(1,0) = 0/1 = 0 with no samples.
    assert.equal(row.accuracy, 0);
  }
});

// T8: single-threshold list — calibrator must select it unconditionally.
test("threshold calibrator with single threshold returns that threshold", async () => {
  const classifier = new FunctionClassifier(() => ["unsafe", 0.4], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });
  const calibrator = new ThresholdCalibrator(jury);

  const threshold = await calibrator.calibrate({
    texts: ["a", "b"],
    labels: ["safe", "unsafe"],
    thresholds: [0.42],
  });

  assert.equal(threshold, 0.42);
  assert.equal(jury.threshold, 0.42);
  assert.equal(calibrator.calibrationReport().rows.length, 1);
});

// T8: NaN confidence — `NaN < threshold` is false, so the sample is treated as
// a non-escalation and counted by label-match. Pinning this so a future
// "throw on NaN" change is an explicit decision rather than a silent regression.
test("threshold calibrator treats NaN confidence as non-escalation", async () => {
  const values: Record<string, [string, number]> = {
    a: ["safe", Number.NaN],
    b: ["unsafe", Number.NaN],
  };
  const classifier = new FunctionClassifier((text: string) => values[text], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });
  const calibrator = new ThresholdCalibrator(jury);

  const threshold = await calibrator.calibrate({
    texts: ["a", "b"],
    labels: ["safe", "unsafe"],
    thresholds: [0.5],
    errorCost: 10,
    escalationCost: 0.05,
  });

  const report = calibrator.calibrationReport();
  const row = report.rows[0]!;
  assert.equal(threshold, 0.5);
  // Both labels match → 2 correct, 0 errors, 0 escalations.
  assert.equal(row.escalationRate, 0);
  assert.equal(row.accuracy, 1);
  assert.equal(row.totalCost, 0);
});
