import test from "node:test";
import assert from "node:assert/strict";

import type { ClassificationResult, Classifier } from "../../src/classifiers/base.ts";
import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { JuryEvaluator, TooManyEscalationsError } from "../../src/evaluation/evaluator.ts";
import type { EvaluateOptions, EvaluationReport } from "../../src/evaluation/evaluator.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { Jury } from "../../src/jury/core.ts";
import type { LLMClient } from "../../src/llm/client.ts";
import type { Persona } from "../../src/personas/base.ts";
import { FakeLLMClient } from "../helpers.ts";
import { CountingClassifier, EXPECTED, FIXTURE, LABELS, SWEEP_THRESHOLDS, ScriptedJury, TEXTS } from "./fixture.ts";
import type { FixtureRow } from "./fixture.ts";

function close(actual: number | null | undefined, expected: number, message?: string): void {
  assert.ok(typeof actual === "number", message ?? `expected a number, got ${String(actual)}`);
  assert.ok(Math.abs(actual - expected) < 1e-9, message ?? `expected ${expected}, got ${actual}`);
}

async function evaluate(jury: ScriptedJury, options: Partial<EvaluateOptions> = {}): Promise<EvaluationReport> {
  return new JuryEvaluator(jury).evaluate({ texts: TEXTS, labels: EXPECTED, ...options });
}

// --- shared fixture: same numbers as the Python tests ------------------------

test("summary matches hand-computed values", async () => {
  const summary = (await evaluate(new ScriptedJury())).summary();

  assert.equal(summary.n, 10);
  assert.equal(summary.bandUpper, 0.95);
  close(summary.primaryAccuracy, 0.4);
  assert.equal(summary.debated, 8);
  close(summary.juryAccuracyOnDebated, 0.75);
  close(summary.primaryAccuracyOnDebated, 0.375);
  assert.equal(summary.flipsHelped, 4);
  assert.equal(summary.flipsHurt, 1);
  close(summary.debateCostUsd, 1.75);
  assert.equal(summary.unpricedCalls, 3);
  close(summary.meanDebateCostUsd, 0.25);
  assert.equal(summary.latencyMsP50, 400);
  assert.equal(summary.latencyMsP95, 800);
  assert.equal(summary.degraded, 2);
  assert.deepEqual(summary.fallbacks, {
    llm_judge_fallback_error: 1,
    llm_judge_fallback_personas_failed: 1,
  });
  assert.deepEqual(summary.confusion, {
    primary: {
      safe: { safe: 2, unsafe: 3 },
      unsafe: { safe: 3, unsafe: 2 },
    },
    jury: {
      safe: { safe: 3, unsafe: 1 },
      unsafe: { safe: 1, unsafe: 3 },
    },
  });
});

test("items record primary and jury outcomes", async () => {
  const report = await evaluate(new ScriptedJury());
  const byText = new Map(report.items.map((item) => [item.text, item]));

  const top = byText.get("t1")!;
  assert.equal(top.debated, false);
  assert.equal(top.juryLabel, null);
  assert.equal(top.primaryCorrect, true);
  assert.equal(top.primaryCostUsd, 0);

  const helped = byText.get("t3")!;
  assert.equal(helped.debated, true);
  assert.equal(helped.primaryCorrect, false);
  assert.equal(helped.juryCorrect, true);
  assert.equal(helped.juryLabel, "unsafe");
  assert.equal(helped.juryCostUsd, 0.25);
  assert.equal(helped.juryDurationMs, 100);
  assert.equal(helped.juryStrategy, "llm_judge");

  const unpriced = byText.get("t7")!;
  assert.equal(unpriced.juryCostUsd, null);
  assert.equal(unpriced.unpricedCalls, 3);
  assert.equal(unpriced.juryDegraded, true);
});

test("threshold sweep matches hand-computed values", async () => {
  const report = await evaluate(new ScriptedJury());
  const rows = report.thresholdSweep({ thresholds: SWEEP_THRESHOLDS, errorCost: 10 });

  // [threshold, escalationRate, systemAccuracy, juryAccuracy, primaryAccuracy,
  //  errors, totalCost]; escalation cost is the measured mean debate cost, 0.25.
  const expected: Array<[number, number, number, number, number, number, number]> = [
    [0.5, 0.2, 0.6, 1.0, 4 / 8, 4, 40.5],
    [0.6, 0.3, 0.6, 1.0, 3 / 7, 4, 40.75],
    [0.7, 0.4, 0.6, 3 / 4, 3 / 6, 4, 41.0],
    [0.8, 0.6, 0.6, 4 / 6, 2 / 4, 4, 41.5],
    [0.9, 0.7, 0.6, 5 / 7, 1 / 3, 4, 41.75],
    [0.95, 0.8, 0.7, 6 / 8, 1 / 2, 3, 32.0],
  ];
  assert.equal(rows.length, expected.length);
  rows.forEach((row, idx) => {
    const want = expected[idx]!;
    const label = `threshold ${want[0]}`;
    assert.equal(row.threshold, want[0], label);
    close(row.escalationRate, want[1], label);
    close(row.systemAccuracy, want[2], label);
    close(row.juryAccuracy, want[3], label);
    close(row.primaryAccuracy, want[4], label);
    assert.equal(row.errors, want[5], label);
    close(row.totalCost, want[6], label);
  });
});

test("best threshold matches hand-computed values", async () => {
  const report = await evaluate(new ScriptedJury());
  assert.equal(report.bestThreshold({ errorCost: 10, thresholds: SWEEP_THRESHOLDS }), 0.95);
  // Errors cheap, escalations expensive: escalate as little as possible.
  assert.equal(report.bestThreshold({ errorCost: 1, escalationCost: 1, thresholds: SWEEP_THRESHOLDS }), 0.5);
});

test("an always-wrong jury picks the lowest threshold", async () => {
  const report = await evaluate(new ScriptedJury("wrong"));
  assert.equal(report.bestThreshold({ escalationCost: 0.01 }), 0.5);
  assert.equal(report.bestThreshold(), 0.5);
});

test("an always-right jury with cheap escalations picks the highest threshold", async () => {
  const report = await evaluate(new ScriptedJury("right"));
  assert.equal(report.bestThreshold({ escalationCost: 0.01 }), 0.95);
});

test("toDict uses the Python SDK's snake_case keys", async () => {
  const report = await evaluate(new ScriptedJury());
  const data = JSON.parse(JSON.stringify(report.toDict()));
  assert.equal(data.band_upper, 0.95);
  assert.equal(data.summary.flips_helped, 4);
  assert.equal(data.summary.mean_debate_cost_usd, 0.25);
  assert.deepEqual(data.summary.confusion.jury.safe, { safe: 3, unsafe: 1 });
  assert.equal(data.items.length, 10);
  assert.equal(data.items[6].jury_cost_usd, null);
  assert.deepEqual(Object.keys(data.items[0]), [
    "text",
    "expected",
    "primary_label",
    "primary_confidence",
    "primary_correct",
    "primary_cost_usd",
    "debated",
    "jury_label",
    "jury_confidence",
    "jury_correct",
    "jury_strategy",
    "jury_cost_usd",
    "jury_duration_ms",
    "jury_degraded",
    "unpriced_calls",
  ]);
});

// --- routing -------------------------------------------------------------------

test("the primary classifier runs exactly once per text", async () => {
  const jury = new ScriptedJury();
  await evaluate(jury);
  assert.deepEqual(jury.classifier.calls, Object.fromEntries(TEXTS.map((text) => [text, 1])));
});

test("only items below bandUpper are debated", async () => {
  const jury = new ScriptedJury();
  await evaluate(jury, { bandUpper: 0.7 });
  // Confidences below 0.7: t7 (0.60), t8, t9, t10. t6 is exactly 0.70.
  assert.deepEqual([...jury.escalated].sort(), ["t10", "t7", "t8", "t9"]);
});

test("maxEscalations throws before any debate", async () => {
  const jury = new ScriptedJury();
  await assert.rejects(
    evaluate(jury, { maxEscalations: 7 }),
    (err: unknown) => err instanceof TooManyEscalationsError && /8 item/.test(err.message),
  );
  assert.deepEqual(jury.escalated, []);

  // Exactly at the cap is allowed.
  const report = await evaluate(new ScriptedJury(), { maxEscalations: 8 });
  assert.equal(report.summary().debated, 8);
});

const personas: Persona[] = [0, 1, 2].map((i) => ({
  name: `P${i}`,
  role: "r",
  systemPrompt: `P${i}`,
  model: "persona-model",
  temperature: 0,
}));

function llmJury(client: LLMClient, confidence: number, classifier?: Classifier): Jury {
  return new Jury({
    classifier: classifier ?? new FunctionClassifier(() => ["safe", confidence], LABELS),
    personas,
    judge: new MajorityVoteJudge(),
    llmClient: client,
  });
}

test("maxEscalations throws before any LLM call", async () => {
  const client = new FakeLLMClient();
  const jury = llmJury(client, 0.5);
  await assert.rejects(
    new JuryEvaluator(jury).evaluate({ texts: ["a", "b"], labels: LABELS, maxEscalations: 1 }),
    TooManyEscalationsError,
  );
  assert.equal(client.calls.length, 0);
});

test("a non-finite confidence is always debated", async () => {
  const rows: FixtureRow[] = [
    ["n1", "safe", "unsafe", Number.NaN, "safe", 0.25, 0, 0, "llm_judge", 10],
    ["n2", "safe", "safe", 0.99, null, null, 0, 0, null, null],
  ];
  const jury = new ScriptedJury("script", rows);
  const report = await new JuryEvaluator(jury).evaluate({ texts: ["n1", "n2"], labels: ["safe", "safe"] });
  assert.deepEqual(jury.escalated, ["n1"]);
  const row = report.thresholdSweep({ thresholds: [0.5] })[0]!;
  assert.equal(row.errors, 0);
  assert.equal(row.escalationRate, 0.5);
});

test("jury stats are not touched", async () => {
  const jury = llmJury(new FakeLLMClient(), 0.5);
  await new JuryEvaluator(jury).evaluate({ texts: ["a", "b"], labels: ["safe", "unsafe"] });
  assert.deepEqual([jury.stats.total, jury.stats.fastPath, jury.stats.escalated], [0, 0, 0]);
});

test("invalid arguments are rejected", async () => {
  const evaluator = new JuryEvaluator(new ScriptedJury());
  await assert.rejects(evaluator.evaluate({ texts: ["t1"], labels: [] }), /same length/);
  for (const options of [
    { bandUpper: 1.5 },
    { bandUpper: Number.NaN },
    { concurrency: 0 },
    { concurrency: 1.5 },
    { maxEscalations: -1 },
  ]) {
    await assert.rejects(evaluator.evaluate({ texts: TEXTS, labels: EXPECTED, ...options }), RangeError);
  }
});

test("a jury without personas is rejected before classifying", async () => {
  const classifier = new CountingClassifier({ a: ["safe", 0.5] });
  const jury = new Jury({ classifier, personas: [], llmClient: new FakeLLMClient() });
  await assert.rejects(new JuryEvaluator(jury).evaluate({ texts: ["a"], labels: ["safe"] }), /at least one persona/);
  assert.deepEqual(classifier.calls, {});
});

// --- sweep rules ---------------------------------------------------------------

test("thresholds above bandUpper are rejected", async () => {
  const report = await evaluate(new ScriptedJury(), { bandUpper: 0.8 });
  assert.throws(() => report.thresholdSweep({ thresholds: [0.5, 0.9] }), /above bandUpper/);
  assert.throws(() => report.bestThreshold({ thresholds: [0.85] }), /above bandUpper/);
});

test("default thresholds stop at bandUpper", async () => {
  const report = await evaluate(new ScriptedJury(), { bandUpper: 0.8 });
  assert.deepEqual(
    report.thresholdSweep().map((row) => row.threshold),
    [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
  );
});

test("escalation cost defaults to the measured mean", async () => {
  const report = await evaluate(new ScriptedJury());
  close(report.thresholdSweep({ thresholds: [0.5], errorCost: 0 })[0]!.totalCost, 2 * 0.25);
});

test("escalation cost falls back to 0.05 when nothing was priced", async () => {
  const rows = FIXTURE.map((r): FixtureRow => (r[4] ? [r[0], r[1], r[2], r[3], r[4], null, 2, 0, r[8], r[9]] : r));
  const report = await evaluate(new ScriptedJury("script", rows));
  assert.equal(report.summary().meanDebateCostUsd, null);
  close(report.thresholdSweep({ thresholds: [0.5], errorCost: 0 })[0]!.totalCost, 2 * 0.05);
});

test("nothing debated gives empty jury fields", async () => {
  const report = await evaluate(new ScriptedJury(), { bandUpper: 0.3 });
  const summary = report.summary();
  assert.equal(summary.debated, 0);
  assert.equal(summary.juryAccuracyOnDebated, null);
  assert.equal(summary.debateCostUsd, null);
  assert.equal(summary.latencyMsP50, null);
  assert.deepEqual(summary.fallbacks, {});
  const rows = report.thresholdSweep();
  assert.deepEqual(
    rows.map((row) => row.threshold),
    [0.3],
  );
  assert.equal(rows[0]!.juryAccuracy, null);
});

// --- with a real Jury -------------------------------------------------------

/** Personas answer "unsafe"; `costUsd` is omitted when `cost` is undefined. */
class UnsafeClient implements LLMClient {
  calls = 0;
  private cost?: number;
  constructor(cost?: number) {
    this.cost = cost;
  }
  async complete() {
    this.calls += 1;
    const content = JSON.stringify({ label: "unsafe", confidence: 0.9, reasoning: "r", key_factors: [] });
    return this.cost === undefined ? { content, tokens: 10 } : { content, tokens: 10, costUsd: this.cost };
  }
}

test("unknown debate cost is null with unpriced calls, never 0", async () => {
  const jury = llmJury(new UnsafeClient(), 0.5);
  const report = await new JuryEvaluator(jury).evaluate({ texts: ["a", "b"], labels: ["unsafe", "safe"] });
  const summary = report.summary();

  assert.equal(summary.debated, 2);
  assert.equal(summary.debateCostUsd, null);
  assert.equal(summary.meanDebateCostUsd, null);
  assert.equal(summary.unpricedCalls, 6);
  for (const item of report.items) {
    assert.equal(item.juryCostUsd, null);
    assert.equal(item.unpricedCalls, 3);
  }
  const data = report.toDict() as { summary: Record<string, unknown> };
  assert.equal(JSON.parse(JSON.stringify(data.summary)).debate_cost_usd, null);
});

test("known debate cost leaves out the primary cost", async () => {
  class PricedClassifier implements Classifier {
    labels = LABELS;
    async classify(): Promise<ClassificationResult> {
      return { label: "safe", confidence: 0.5, costUsd: 0.5 };
    }
  }
  const jury = llmJury(new UnsafeClient(0.002), 0.5, new PricedClassifier());
  const report = await new JuryEvaluator(jury).evaluate({ texts: ["a"], labels: ["unsafe"] });
  const item = report.items[0]!;
  assert.equal(item.primaryCostUsd, 0.5);
  close(item.juryCostUsd, 0.006);
  assert.equal(item.juryLabel, "unsafe");
  assert.equal(item.juryCorrect, true);
  assert.equal(report.summary().flipsHelped, 1);
});
