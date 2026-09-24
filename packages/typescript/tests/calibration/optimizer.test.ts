import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { DebateConfig, DebateMode } from "../../src/debate/engine.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { Jury } from "../../src/jury/core.ts";
import { ThresholdCalibrator } from "../../src/calibration/optimizer.ts";
import { CountingClassifier, EXPECTED, SWEEP_THRESHOLDS, ScriptedJury, TEXTS } from "../evaluation/fixture.ts";
import { FakeLLMClient } from "../helpers.ts";

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

// A NaN confidence escalates, the same rule Jury.shouldEscalate uses (BUG-02),
// so the calibrator does not trust a primary label the jury would not.
test("threshold calibrator escalates a NaN confidence like the jury", async () => {
  const values: Record<string, [string, number]> = {
    a: ["safe", Number.NaN],
    b: ["unsafe", 0.9],
  };
  const classifier = new FunctionClassifier((text: string) => values[text]!, ["safe", "unsafe"]);
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
  // a escalates; b is kept and right.
  assert.equal(row.escalationRate, 0.5);
  assert.equal(row.accuracy, 1);
  assert.ok(Math.abs(row.totalCost - 0.05) < 1e-12);
});

// --- FEAT-01: classify once, escalations out of accuracy, measured mode -----

test("cheap mode classifies each text once", async () => {
  const classifier = new CountingClassifier({ a: ["safe", 0.9], b: ["unsafe", 0.4] });
  const jury = new Jury({ classifier, personas: [], llmClient: new FakeLLMClient() });
  await new ThresholdCalibrator(jury).calibrate({
    texts: ["a", "b"],
    labels: ["safe", "unsafe"],
    thresholds: [0.5, 0.6, 0.7, 0.8, 0.9],
  });
  assert.deepEqual(classifier.calls, { a: 1, b: 1 });
});

test("cheap mode leaves escalated items out of accuracy", async () => {
  // Always-wrong primary: escalations must not count as correct.
  const classifier = new CountingClassifier({ a: ["unsafe", 0.4], b: ["safe", 0.6], c: ["safe", 0.9] });
  const jury = new Jury({ classifier, personas: [], llmClient: new FakeLLMClient() });
  const calibrator = new ThresholdCalibrator(jury);
  await calibrator.calibrate({
    texts: ["a", "b", "c"],
    labels: ["safe", "unsafe", "unsafe"],
    escalationCost: 0.05,
    thresholds: [0.5, 0.95],
  });
  const [low, high] = calibrator.calibrationReport().rows;
  // t=0.5: a escalates, b and c are kept and both wrong.
  assert.equal(low!.accuracy, 0);
  assert.ok(Math.abs(low!.escalationRate - 1 / 3) < 1e-12);
  assert.ok(Math.abs(low!.totalCost - 20.05) < 1e-9);
  // t=0.95: everything escalates, nothing is resolved.
  assert.equal(high!.accuracy, 0);
  assert.equal(high!.escalationRate, 1);
  assert.ok(Math.abs(high!.totalCost - 0.15) < 1e-12);
  const report = calibrator.calibrationReport();
  assert.equal(report.useJury, false);
  assert.equal(report.summary, undefined);
  assert.deepEqual(Object.keys(report.rows[0]!), ["threshold", "accuracy", "escalationRate", "totalCost"]);
});

test("cheap mode escalation cost defaults to five cents", async () => {
  const classifier = new CountingClassifier({ a: ["safe", 0.4] });
  const jury = new Jury({ classifier, personas: [], llmClient: new FakeLLMClient() });
  const calibrator = new ThresholdCalibrator(jury);
  await calibrator.calibrate({ texts: ["a"], labels: ["safe"], thresholds: [0.5] });
  assert.ok(Math.abs(calibrator.calibrationReport().rows[0]!.totalCost - 0.05) < 1e-12);
});

test("useJury rows come from the measured sweep", async () => {
  const jury = new ScriptedJury();
  const calibrator = new ThresholdCalibrator(jury);
  const best = await calibrator.calibrate({
    texts: TEXTS,
    labels: EXPECTED,
    thresholds: SWEEP_THRESHOLDS,
    useJury: true,
  });

  assert.equal(best, 0.95);
  assert.equal(jury.threshold, 0.95);
  const report = calibrator.calibrationReport();
  assert.equal(report.useJury, true);
  assert.equal(report.bestThreshold, 0.95);
  assert.equal(report.summary?.flipsHelped, 4);

  const first = report.rows[0]!;
  assert.equal(first.threshold, 0.5);
  assert.ok(Math.abs(first.accuracy - 0.5) < 1e-12); // primary, kept items only
  assert.ok(Math.abs((first.systemAccuracy ?? -1) - 0.6) < 1e-12);
  assert.equal(first.juryAccuracy, 1);
  assert.ok(Math.abs(first.escalationRate - 0.2) < 1e-12);
  assert.ok(Math.abs(first.totalCost - 40.5) < 1e-9);
  const last = report.rows[report.rows.length - 1]!;
  assert.ok(Math.abs((last.systemAccuracy ?? -1) - 0.7) < 1e-12);
  assert.ok(Math.abs(last.totalCost - 32) < 1e-9);
  assert.equal(calibrator.evaluationReport?.bandUpper, 0.95);
});

test("useJury classifies each text once", async () => {
  const jury = new ScriptedJury();
  await new ThresholdCalibrator(jury).calibrate({ texts: TEXTS, labels: EXPECTED, useJury: true });
  assert.deepEqual(jury.classifier.calls, Object.fromEntries(TEXTS.map((text) => [text, 1])));
});

test("useJury debates only below the highest threshold", async () => {
  const jury = new ScriptedJury();
  await new ThresholdCalibrator(jury).calibrate({
    texts: TEXTS,
    labels: EXPECTED,
    thresholds: [0.5, 0.6],
    useJury: true,
  });
  assert.deepEqual([...jury.escalated].sort(), ["t10", "t8", "t9"]);
});

test("useJury runs the real jury with a fake client", async () => {
  const client = new FakeLLMClient({
    persona: {
      content: JSON.stringify({ label: "unsafe", confidence: 0.9, reasoning: "r", key_factors: [] }),
      costUsd: 0.001,
    },
  });
  const classifier = new CountingClassifier({ a: ["safe", 0.6], b: ["safe", 0.99] });
  const jury = new Jury({
    classifier,
    personas: [{ name: "A", role: "r", systemPrompt: "A", model: "persona", temperature: 0 }],
    judge: new MajorityVoteJudge(),
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
    llmClient: client,
  });
  const calibrator = new ThresholdCalibrator(jury);
  const best = await calibrator.calibrate({
    texts: ["a", "b"],
    labels: ["unsafe", "safe"],
    thresholds: [0.5, 0.7],
    useJury: true,
  });
  // Item a is wrong at 0.5 and fixed by the jury at 0.7.
  assert.equal(best, 0.7);
  assert.deepEqual(classifier.calls, { a: 1, b: 1 });
  assert.equal(client.calls.length, 1);
  assert.equal(jury.stats.total, 0);
});
