/**
 * Shared evaluation fixture. packages/python/tests/test_evaluation/fixture.py
 * holds the same rows, and both SDKs' tests assert the same hand-computed
 * numbers against them.
 */
import type { ClassificationResult, Classifier } from "../../src/classifiers/base.ts";
import type { DebateTranscript } from "../../src/debate/engine.ts";
import { Verdict } from "../../src/judges/base.ts";

export const LABELS = ["safe", "unsafe"];

export type FixtureRow = [
  text: string,
  expected: string,
  primaryLabel: string,
  primaryConfidence: number,
  juryLabel: string | null,
  juryCost: number | null,
  unpricedCalls: number,
  personaFailures: number,
  strategy: string | null,
  durationMs: number | null,
];

const JUDGED = "llm_judge";
const ERROR = "llm_judge_fallback_error";
const ALL_FAILED = "llm_judge_fallback_personas_failed";

// Items at or above bandUpper=0.95 have no jury script.
export const FIXTURE: FixtureRow[] = [
  ["t1", "safe", "safe", 0.98, null, null, 0, 0, null, null],
  ["t2", "unsafe", "safe", 0.96, null, null, 0, 0, null, null],
  ["t3", "unsafe", "safe", 0.92, "unsafe", 0.25, 0, 0, JUDGED, 100],
  ["t4", "safe", "safe", 0.85, "safe", 0.25, 0, 0, JUDGED, 200],
  ["t5", "safe", "unsafe", 0.75, "safe", 0.25, 0, 1, JUDGED, 300],
  ["t6", "unsafe", "unsafe", 0.7, "safe", 0.25, 0, 0, ERROR, 400],
  ["t7", "safe", "unsafe", 0.6, "unsafe", null, 3, 3, ALL_FAILED, 500],
  ["t8", "unsafe", "unsafe", 0.55, "unsafe", 0.25, 0, 0, JUDGED, 600],
  ["t9", "safe", "unsafe", 0.45, "safe", 0.25, 0, 0, JUDGED, 700],
  ["t10", "unsafe", "safe", 0.4, "unsafe", 0.25, 0, 0, JUDGED, 800],
];
export const TEXTS = FIXTURE.map((row) => row[0]);
export const EXPECTED = FIXTURE.map((row) => row[1]);
export const SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95];

/** Replays stored predictions and counts calls per text. */
export class CountingClassifier implements Classifier {
  labels = LABELS;
  calls: Record<string, number> = {};
  private predictions: Record<string, [string, number]>;

  constructor(predictions: Record<string, [string, number]>) {
    this.predictions = predictions;
  }

  async classify(text: string): Promise<ClassificationResult> {
    this.calls[text] = (this.calls[text] ?? 0) + 1;
    const [label, confidence] = this.predictions[text]!;
    return { label, confidence, costUsd: 0 };
  }
}

function other(label: string): string {
  return label === "safe" ? "unsafe" : "safe";
}

/**
 * A jury whose escalations return scripted verdicts. Mode "script" uses the
 * fixture's jury columns; "right" and "wrong" answer the expected label or
 * the other one for every item.
 */
export class ScriptedJury {
  classifier: CountingClassifier;
  personas = ["p"];
  threshold = 0.7;
  escalated: string[] = [];
  private rows: Map<string, FixtureRow>;
  private mode: "script" | "right" | "wrong";

  constructor(mode: "script" | "right" | "wrong" = "script", rows: FixtureRow[] = FIXTURE) {
    this.classifier = new CountingClassifier(Object.fromEntries(rows.map((r) => [r[0], [r[2], r[3]]])));
    this.rows = new Map(rows.map((r) => [r[0], r]));
    this.mode = mode;
  }

  async escalate(text: string, primary: ClassificationResult): Promise<Verdict> {
    this.escalated.push(text);
    const row = this.rows.get(text)!;
    const expected = row[1];
    let [, , , , label, cost, unpriced, failures] = row;
    const strategy = row[8] ?? JUDGED;
    const duration = row[9] ?? 0;
    if (this.mode === "right") {
      [label, cost, unpriced, failures] = [expected, 0.25, 0, 0];
    } else if (this.mode === "wrong") {
      [label, cost, unpriced, failures] = [other(expected), 0.25, 0, 0];
    }
    const transcript: DebateTranscript = {
      inputText: text,
      primaryResult: primary,
      rounds: [],
      durationMs: duration,
      totalTokens: 0,
      totalCostUsd: cost,
      unpricedCalls: unpriced,
    };
    return new Verdict({
      label: label!,
      confidence: 0.8,
      reasoning: "scripted",
      wasEscalated: true,
      primaryResult: primary,
      debateTranscript: transcript,
      judgeStrategy: strategy,
      totalDurationMs: duration,
      totalCostUsd: cost === null ? null : (primary.costUsd ?? 0) + cost,
      personaFailures: failures,
    });
  }
}
