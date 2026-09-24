import type { ClassificationResult } from "../classifiers/base.ts";
import {
  DEFAULT_ESCALATION_COST_USD,
  DEFAULT_THRESHOLDS,
  JuryEvaluator,
} from "../evaluation/evaluator.ts";
import type { EvaluableJury, EvaluationReport, EvaluationSummary } from "../evaluation/evaluator.ts";

export type CalibrationOptions = {
  texts: string[];
  labels: string[];
  errorCost?: number;
  /**
   * Cost of one escalation. Default 0.05; with `useJury`, the measured mean
   * debate cost (0.05 when no debate reported a cost).
   */
  escalationCost?: number | null;
  thresholds?: number[];
  /**
   * Run the jury on every item below the highest threshold and pick the
   * threshold from its measured outcomes. Default false.
   */
  useJury?: boolean;
};

export type CalibrationRow = {
  threshold: number;
  /** Primary accuracy on the items that do not escalate (0 when all do). */
  accuracy: number;
  escalationRate: number;
  totalCost: number;
  /** Only with `useJury`. */
  systemAccuracy?: number | null;
  /** Only with `useJury`. */
  juryAccuracy?: number | null;
};

export type CalibrationReport = {
  bestThreshold: number | null;
  useJury: boolean;
  rows: CalibrationRow[];
  /** The evaluation summary, only after a `useJury` calibration. */
  summary?: EvaluationSummary;
};

/** A jury the calibrator can tune: one it can evaluate and set a threshold on. */
export type CalibratableJury = EvaluableJury & { threshold: number };

/** Jury routing: a non-finite confidence always escalates. */
function escalates(result: ClassificationResult, threshold: number): boolean {
  return typeof result.confidence !== "number" || !Number.isFinite(result.confidence) || result.confidence < threshold;
}

/**
 * Finds the confidence threshold with the lowest expected cost.
 *
 * Classifies each text once, then sweeps threshold candidates over the cached
 * `(label, confidence)` pairs. An item escalates at threshold `t` when its
 * confidence is below `t` or not finite, the same rule `Jury` uses.
 *
 * Without `useJury` the jury never runs: escalations are priced at
 * `escalationCost` and left out of `accuracy`, and nothing is assumed about
 * whether the jury would get them right. With `useJury` it runs
 * `JuryEvaluator` (one debate per item below the highest threshold) and picks
 * the threshold from the measured outcomes.
 */
export class ThresholdCalibrator {
  private jury: CalibratableJury;
  private rows: CalibrationRow[] = [];
  private bestThreshold: number | null = null;
  private useJury = false;
  /** The jury measurement behind the last `useJury` calibration. */
  evaluationReport: EvaluationReport | null = null;

  constructor(jury: CalibratableJury) {
    this.jury = jury;
  }

  /**
   * Pick the threshold that minimises `errors * errorCost + escalations *
   * escalationCost` and set it on the jury. With `useJury` errors count the
   * jury's label for escalated items; otherwise escalated items count no error.
   */
  async calibrate(options: CalibrationOptions): Promise<number> {
    if (options.texts.length !== options.labels.length) {
      throw new Error("texts and labels must have same length");
    }
    const errorCost = options.errorCost ?? 10;
    const thresholds =
      options.thresholds && options.thresholds.length > 0 ? [...options.thresholds] : [...DEFAULT_THRESHOLDS];
    const useJury = options.useJury ?? false;

    const rows = useJury
      ? await this.measuredRows(options, thresholds, errorCost)
      : await this.cheapRows(options, thresholds, errorCost, options.escalationCost ?? DEFAULT_ESCALATION_COST_USD);

    let best = rows[0]!;
    for (const row of rows.slice(1)) {
      if (row.totalCost < best.totalCost) {
        best = row;
      }
    }

    this.rows = rows;
    this.useJury = useJury;
    this.bestThreshold = best.threshold;
    this.jury.threshold = best.threshold;
    return best.threshold;
  }

  private async cheapRows(
    options: CalibrationOptions,
    thresholds: number[],
    errorCost: number,
    escalationCost: number,
  ): Promise<CalibrationRow[]> {
    this.evaluationReport = null;
    // Classify each text once and cache the results.
    const cached: ClassificationResult[] = [];
    for (const text of options.texts) {
      cached.push(await this.jury.classifier.classify(text));
    }

    const total = Math.max(1, options.texts.length);
    return thresholds.map((threshold) => {
      let errors = 0;
      let escalations = 0;
      let correct = 0;
      cached.forEach((result, idx) => {
        if (escalates(result, threshold)) {
          // The item would be escalated. Without running the jury its outcome
          // is unknown, so it counts as neither right nor wrong.
          escalations += 1;
        } else if (result.label === options.labels[idx]) {
          correct += 1;
        } else {
          errors += 1;
        }
      });
      const resolved = correct + errors;
      return {
        threshold,
        accuracy: resolved > 0 ? correct / resolved : 0,
        escalationRate: escalations / total,
        totalCost: errors * errorCost + escalations * escalationCost,
      };
    });
  }

  private async measuredRows(
    options: CalibrationOptions,
    thresholds: number[],
    errorCost: number,
  ): Promise<CalibrationRow[]> {
    const report = await new JuryEvaluator(this.jury).evaluate({
      texts: options.texts,
      labels: options.labels,
      bandUpper: Math.max(...thresholds),
    });
    this.evaluationReport = report;
    return report
      .thresholdSweep({ thresholds, errorCost, escalationCost: options.escalationCost ?? null })
      .map((row) => ({
        threshold: row.threshold,
        accuracy: row.primaryAccuracy ?? 0,
        escalationRate: row.escalationRate ?? 0,
        totalCost: row.totalCost,
        systemAccuracy: row.systemAccuracy,
        juryAccuracy: row.juryAccuracy,
      }));
  }

  /**
   * The last calibration: best threshold and one row per candidate. Rows
   * carry `threshold`, `accuracy` (primary accuracy on the items that do not
   * escalate), `escalationRate` and `totalCost`. After a `useJury`
   * calibration they also carry `systemAccuracy` and `juryAccuracy`, and the
   * report adds the evaluation `summary`.
   */
  calibrationReport(): CalibrationReport {
    const report: CalibrationReport = {
      bestThreshold: this.bestThreshold,
      useJury: this.useJury,
      rows: this.rows.map((row) => ({ ...row })),
    };
    if (this.useJury && this.evaluationReport) {
      report.summary = this.evaluationReport.summary();
    }
    return report;
  }
}
