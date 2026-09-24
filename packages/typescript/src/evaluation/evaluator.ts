/**
 * Measure what the jury does on labelled data.
 *
 * `JuryEvaluator` classifies every text once with the jury's primary
 * classifier, sends the items whose confidence is below `bandUpper` to
 * `Jury.escalate`, and returns an `EvaluationReport`. The report compares the
 * jury's labels with the primary labels and sweeps escalation thresholds over
 * the measured outcomes, so the threshold is chosen from what the jury
 * actually did.
 */
import type { ClassificationResult, Classifier } from "../classifiers/base.ts";
import type { Verdict } from "../judges/base.ts";
import { createSemaphore } from "../utils.ts";

/** Used by `thresholdSweep` when `escalationCost` is unset and no debate reported a cost. */
export const DEFAULT_ESCALATION_COST_USD = 0.05;

/** The ThresholdCalibrator grid: 0.5, 0.55, ..., 0.95. */
export const DEFAULT_THRESHOLDS: readonly number[] = Array.from({ length: 10 }, (_v, idx) =>
  Number((0.5 + idx * 0.05).toFixed(2)),
);

/** More items fall below `bandUpper` than `maxEscalations` allows. */
export class TooManyEscalationsError extends RangeError {
  constructor(message: string) {
    super(message);
    this.name = "TooManyEscalationsError";
  }
}

/** What `JuryEvaluator` needs from a jury. `Jury` satisfies it. */
export type EvaluableJury = {
  classifier: Classifier;
  personas?: readonly unknown[];
  escalate(text: string, primary: ClassificationResult): Promise<Verdict>;
};

export type EvaluateOptions = {
  texts: string[];
  labels: string[];
  /** Debate every item whose primary confidence is below this. Default 0.95. */
  bandUpper?: number;
  /** Throw before any debate when more items than this would be debated. */
  maxEscalations?: number | null;
  /** Calls in flight in each pass. Default 5. */
  concurrency?: number;
};

/** One labelled text: the primary result and, if debated, the jury's. */
export type EvaluationItem = {
  text: string;
  expected: string;
  primaryLabel: string;
  primaryConfidence: number;
  primaryCorrect: boolean;
  primaryCostUsd: number | null;
  debated: boolean;
  juryLabel: string | null;
  juryConfidence: number | null;
  juryCorrect: boolean | null;
  juryStrategy: string | null;
  /** Debate and judge cost: the verdict total minus the primary cost. Null when either is unknown. */
  juryCostUsd: number | null;
  juryDurationMs: number | null;
  juryDegraded: boolean | null;
  /** Debate calls that reported no cost (a known juryCostUsd is then a lower bound). */
  unpricedCalls: number;
};

/** `{ expected: { predicted: count } }` */
export type ConfusionMatrix = Record<string, Record<string, number>>;

export type EvaluationSummary = {
  n: number;
  bandUpper: number;
  primaryAccuracy: number | null;
  debated: number;
  juryAccuracyOnDebated: number | null;
  primaryAccuracyOnDebated: number | null;
  flipsHelped: number;
  flipsHurt: number;
  debateCostUsd: number | null;
  unpricedCalls: number;
  meanDebateCostUsd: number | null;
  latencyMsP50: number | null;
  latencyMsP95: number | null;
  degraded: number;
  fallbacks: Record<string, number>;
  confusion: { primary: ConfusionMatrix; jury: ConfusionMatrix };
};

export type ThresholdSweepRow = {
  threshold: number;
  escalationRate: number | null;
  systemAccuracy: number | null;
  juryAccuracy: number | null;
  primaryAccuracy: number | null;
  errors: number;
  totalCost: number;
};

export type ThresholdSweepOptions = {
  /** Default: the calibrator grid (0.5 to 0.95) up to `bandUpper`. */
  thresholds?: number[];
  /** Default 10. */
  errorCost?: number;
  /** Default: the measured mean debate cost, or 0.05 when no debate was priced. */
  escalationCost?: number | null;
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Jury routing: a non-finite confidence always escalates. */
function escalates(confidence: unknown, threshold: number): boolean {
  return !isFiniteNumber(confidence) || confidence < threshold;
}

function ratio(numerator: number, denominator: number): number | null {
  return denominator ? numerator / denominator : null;
}

/** Nearest-rank percentile: an observed value, never an interpolation. */
function percentile(values: number[], percent: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const ordered = [...values].sort((a, b) => a - b);
  const rank = Math.max(1, Math.ceil((percent / 100) * ordered.length));
  return ordered[rank - 1]!;
}

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function checkUnitInterval(name: string, value: unknown): number {
  if (!isFiniteNumber(value) || value < 0 || value > 1) {
    throw new RangeError(`${name} must be a finite number in [0, 1], got ${String(value)}`);
  }
  return value;
}

function checkCost(name: string, value: unknown): number {
  if (!isFiniteNumber(value) || value < 0) {
    throw new RangeError(`${name} must be a finite number >= 0, got ${String(value)}`);
  }
  return value;
}

/**
 * Cost the escalation added on top of the primary call. An escalated
 * verdict's total includes the primary classifier's cost, so the debate part
 * is the difference. Unknown when either side is unknown.
 */
function debateCost(verdictCost: number | null | undefined, primaryCost: number | null | undefined): number | null {
  if (verdictCost == null || primaryCost == null) {
    return null;
  }
  return verdictCost - primaryCost;
}

function summaryToDict(summary: EvaluationSummary): Record<string, unknown> {
  return {
    n: summary.n,
    band_upper: summary.bandUpper,
    primary_accuracy: summary.primaryAccuracy,
    debated: summary.debated,
    jury_accuracy_on_debated: summary.juryAccuracyOnDebated,
    primary_accuracy_on_debated: summary.primaryAccuracyOnDebated,
    flips_helped: summary.flipsHelped,
    flips_hurt: summary.flipsHurt,
    debate_cost_usd: summary.debateCostUsd,
    unpriced_calls: summary.unpricedCalls,
    mean_debate_cost_usd: summary.meanDebateCostUsd,
    latency_ms_p50: summary.latencyMsP50,
    latency_ms_p95: summary.latencyMsP95,
    degraded: summary.degraded,
    fallbacks: summary.fallbacks,
    confusion: summary.confusion,
  };
}

function itemToDict(item: EvaluationItem): Record<string, unknown> {
  return {
    text: item.text,
    expected: item.expected,
    primary_label: item.primaryLabel,
    primary_confidence: item.primaryConfidence,
    primary_correct: item.primaryCorrect,
    primary_cost_usd: item.primaryCostUsd,
    debated: item.debated,
    jury_label: item.juryLabel,
    jury_confidence: item.juryConfidence,
    jury_correct: item.juryCorrect,
    jury_strategy: item.juryStrategy,
    jury_cost_usd: item.juryCostUsd,
    jury_duration_ms: item.juryDurationMs,
    jury_degraded: item.juryDegraded,
    unpriced_calls: item.unpricedCalls,
  };
}

/** A sweep row with the Python SDK's snake_case keys. */
export function sweepRowToDict(row: ThresholdSweepRow): Record<string, unknown> {
  return {
    threshold: row.threshold,
    escalation_rate: row.escalationRate,
    system_accuracy: row.systemAccuracy,
    jury_accuracy: row.juryAccuracy,
    primary_accuracy: row.primaryAccuracy,
    errors: row.errors,
    total_cost: row.totalCost,
  };
}

/** Measured primary and jury outcomes for a labelled dataset. */
export class EvaluationReport {
  readonly items: EvaluationItem[];
  readonly bandUpper: number;

  constructor(items: EvaluationItem[], bandUpper: number) {
    this.items = items;
    this.bandUpper = bandUpper;
  }

  /**
   * Headline numbers for the evaluated dataset. Accuracy fields are null when
   * their denominator is zero. `debateCostUsd` sums the debates whose cost is
   * known and is null when none is; `unpricedCalls` counts debate calls that
   * reported no cost (a non-zero count makes the sum a lower bound).
   * Latencies are nearest-rank percentiles of the debated items'
   * `Jury.escalate` duration.
   */
  summary(): EvaluationSummary {
    const items = this.items;
    const debated = items.filter((item) => item.debated);
    const knownCosts = this.knownDebateCosts();
    const durations = debated
      .map((item) => item.juryDurationMs)
      .filter((value): value is number => value !== null);
    const counts = new Map<string, number>();
    for (const item of debated) {
      const strategy = item.juryStrategy ?? "";
      if (strategy.includes("fallback") || strategy.startsWith("cost_guard")) {
        counts.set(strategy, (counts.get(strategy) ?? 0) + 1);
      }
    }
    const fallbacks: Record<string, number> = {};
    for (const key of [...counts.keys()].sort(compareStrings)) {
      fallbacks[key] = counts.get(key)!;
    }
    const knownSum = knownCosts.reduce((sum, cost) => sum + cost, 0);

    return {
      n: items.length,
      bandUpper: this.bandUpper,
      primaryAccuracy: ratio(items.filter((item) => item.primaryCorrect).length, items.length),
      debated: debated.length,
      juryAccuracyOnDebated: ratio(debated.filter((item) => item.juryCorrect).length, debated.length),
      primaryAccuracyOnDebated: ratio(debated.filter((item) => item.primaryCorrect).length, debated.length),
      flipsHelped: debated.filter((item) => item.juryCorrect && !item.primaryCorrect).length,
      flipsHurt: debated.filter((item) => item.primaryCorrect && !item.juryCorrect).length,
      debateCostUsd: knownCosts.length > 0 ? knownSum : null,
      unpricedCalls: debated.reduce((sum, item) => sum + item.unpricedCalls, 0),
      meanDebateCostUsd: knownCosts.length > 0 ? knownSum / knownCosts.length : null,
      latencyMsP50: percentile(durations, 50),
      latencyMsP95: percentile(durations, 95),
      degraded: debated.filter((item) => item.juryDegraded).length,
      fallbacks,
      confusion: this.confusion(),
    };
  }

  /**
   * Both matrices cover every label seen in the data, sorted. The primary
   * matrix counts all items; the jury matrix counts debated items only.
   */
  private confusion(): { primary: ConfusionMatrix; jury: ConfusionMatrix } {
    const seen = new Set<string>();
    for (const item of this.items) {
      seen.add(item.expected);
      seen.add(item.primaryLabel);
      if (item.debated && item.juryLabel !== null) {
        seen.add(item.juryLabel);
      }
    }
    const labels = [...seen].sort(compareStrings);
    const empty = (): ConfusionMatrix =>
      Object.fromEntries(labels.map((row) => [row, Object.fromEntries(labels.map((col) => [col, 0]))]));
    const primary = empty();
    const jury = empty();
    for (const item of this.items) {
      primary[item.expected]![item.primaryLabel]! += 1;
      if (item.debated && item.juryLabel !== null) {
        jury[item.expected]![item.juryLabel]! += 1;
      }
    }
    return { primary, jury };
  }

  private knownDebateCosts(): number[] {
    return this.items
      .filter((item) => item.debated && item.juryCostUsd !== null)
      .map((item) => item.juryCostUsd as number);
  }

  /** Measured mean debate cost, or 0.05 USD when no debate was priced. */
  private defaultEscalationCost(): number {
    const known = this.knownDebateCosts();
    if (known.length === 0) {
      return DEFAULT_ESCALATION_COST_USD;
    }
    return known.reduce((sum, cost) => sum + cost, 0) / known.length;
  }

  /** The calibrator grid (0.5 to 0.95) up to `bandUpper`. */
  private defaultThresholds(): number[] {
    const candidates = DEFAULT_THRESHOLDS.filter((threshold) => threshold <= this.bandUpper);
    return candidates.length > 0 ? candidates : [this.bandUpper];
  }

  /**
   * Replay the measured outcomes at each threshold. At threshold `t` an item
   * escalates when its primary confidence is below `t` (or not finite) and
   * then takes the jury's label; the others keep the primary label.
   * `totalCost = errors * errorCost + escalations * escalationCost`.
   *
   * `primaryAccuracy` is the primary's accuracy on the items it keeps and
   * `juryAccuracy` the jury's on the items it gets; both are null when their
   * group is empty.
   *
   * Throws for a threshold above `bandUpper`: items at or above `bandUpper`
   * were never debated, so their jury outcome is unknown.
   */
  thresholdSweep(options: ThresholdSweepOptions = {}): ThresholdSweepRow[] {
    const candidates = options.thresholds ? [...options.thresholds] : this.defaultThresholds();
    if (candidates.length === 0) {
      throw new RangeError("at least one threshold is required");
    }
    for (const threshold of candidates) {
      checkUnitInterval("threshold", threshold);
      if (threshold > this.bandUpper) {
        throw new RangeError(
          `threshold ${threshold} is above bandUpper ${this.bandUpper}: items at or above bandUpper ` +
            "were not debated, so their jury outcome was not measured. Re-run evaluate() with a higher bandUpper.",
        );
      }
    }
    const errorCost = checkCost("errorCost", options.errorCost ?? 10);
    const perEscalation =
      options.escalationCost == null
        ? this.defaultEscalationCost()
        : checkCost("escalationCost", options.escalationCost);

    const n = this.items.length;
    return candidates.map((threshold) => {
      let escalations = 0;
      let errors = 0;
      let kept = 0;
      let keptCorrect = 0;
      let juryCorrect = 0;
      for (const item of this.items) {
        if (escalates(item.primaryConfidence, threshold)) {
          escalations += 1;
          if (item.juryCorrect) {
            juryCorrect += 1;
          } else {
            errors += 1;
          }
        } else {
          kept += 1;
          if (item.primaryCorrect) {
            keptCorrect += 1;
          } else {
            errors += 1;
          }
        }
      }
      return {
        threshold,
        escalationRate: ratio(escalations, n),
        systemAccuracy: ratio(n - errors, n),
        juryAccuracy: ratio(juryCorrect, escalations),
        primaryAccuracy: ratio(keptCorrect, kept),
        errors,
        totalCost: errors * errorCost + escalations * perEscalation,
      };
    });
  }

  /** Threshold with the lowest sweep `totalCost`; the first one wins ties. */
  bestThreshold(options: ThresholdSweepOptions = {}): number {
    const rows = this.thresholdSweep(options);
    let best = rows[0]!;
    for (const row of rows.slice(1)) {
      if (row.totalCost < best.totalCost) {
        best = row;
      }
    }
    return best.threshold;
  }

  /** The report with the Python SDK's snake_case keys. */
  toDict(): Record<string, unknown> {
    return {
      band_upper: this.bandUpper,
      summary: summaryToDict(this.summary()),
      items: this.items.map(itemToDict),
    };
  }
}

/**
 * Run async tasks with at most `concurrency` in flight. The first failure
 * rejects, and no new task starts after it, so a failed evaluation stops
 * spending. Tasks already in flight run to completion.
 */
async function runBounded<T>(tasks: Array<() => Promise<T>>, concurrency: number): Promise<T[]> {
  const semaphore = createSemaphore(concurrency);
  let aborted = false;
  return Promise.all(
    tasks.map(async (task) => {
      await semaphore.acquire();
      try {
        if (aborted) {
          // The evaluation has already rejected; this rejection is never observed.
          throw new Error("evaluation aborted after an earlier call failed");
        }
        return await task();
      } catch (err) {
        aborted = true;
        throw err;
      } finally {
        semaphore.release();
      }
    }),
  );
}

/**
 * Runs a jury over labelled data and reports what it changed.
 *
 * The primary classifier is called exactly once per text. Items whose
 * primary confidence is below `bandUpper` (or not finite) go to
 * `jury.escalate`, bypassing the jury's own threshold and
 * `escalationOverride`. `jury.stats` is not touched.
 */
export class JuryEvaluator {
  readonly jury: EvaluableJury;

  constructor(jury: EvaluableJury) {
    this.jury = jury;
  }

  /**
   * Classify every text, debate the ones below `bandUpper`, report.
   *
   * `maxEscalations` caps the number of debates: when more items fall below
   * `bandUpper`, `TooManyEscalationsError` is thrown after the primary pass
   * and before any debate starts. `concurrency` bounds the calls in flight in
   * each pass.
   */
  async evaluate(options: EvaluateOptions): Promise<EvaluationReport> {
    const { texts, labels } = options;
    if (texts.length !== labels.length) {
      throw new Error("texts and labels must have same length");
    }
    const bandUpper = checkUnitInterval("bandUpper", options.bandUpper ?? 0.95);
    const concurrency = options.concurrency ?? 5;
    if (!Number.isInteger(concurrency) || concurrency < 1) {
      throw new RangeError(`concurrency must be an integer >= 1, got ${String(concurrency)}`);
    }
    const maxEscalations = options.maxEscalations ?? null;
    if (maxEscalations !== null && (!Number.isInteger(maxEscalations) || maxEscalations < 0)) {
      throw new RangeError(`maxEscalations must be null or an integer >= 0, got ${String(maxEscalations)}`);
    }
    if (this.jury.personas !== undefined && this.jury.personas.length === 0) {
      throw new Error("JuryEvaluator needs a jury with at least one persona; this jury has none.");
    }

    const classifier = this.jury.classifier;
    const primaries = await runBounded(
      texts.map((text) => () => classifier.classify(text)),
      concurrency,
    );

    const debatedIndexes = primaries.flatMap((primary, index) =>
      escalates(primary.confidence, bandUpper) ? [index] : [],
    );
    if (maxEscalations !== null && debatedIndexes.length > maxEscalations) {
      throw new TooManyEscalationsError(
        `${debatedIndexes.length} item(s) have a primary confidence below bandUpper=${bandUpper}, ` +
          `more than maxEscalations=${maxEscalations}. No debate was run. ` +
          "Raise maxEscalations or lower bandUpper.",
      );
    }

    const verdicts = await runBounded(
      debatedIndexes.map((index) => () => this.jury.escalate(texts[index]!, primaries[index]!)),
      concurrency,
    );
    const verdictByIndex = new Map<number, Verdict>();
    debatedIndexes.forEach((index, position) => verdictByIndex.set(index, verdicts[position]!));

    const items = texts.map((text, index): EvaluationItem => {
      const expected = labels[index]!;
      const primary = primaries[index]!;
      const primaryCostUsd = primary.costUsd ?? null;
      const item: EvaluationItem = {
        text,
        expected,
        primaryLabel: primary.label,
        primaryConfidence: primary.confidence,
        primaryCorrect: primary.label === expected,
        primaryCostUsd,
        debated: false,
        juryLabel: null,
        juryConfidence: null,
        juryCorrect: null,
        juryStrategy: null,
        juryCostUsd: null,
        juryDurationMs: null,
        juryDegraded: null,
        unpricedCalls: 0,
      };
      const verdict = verdictByIndex.get(index);
      if (verdict) {
        item.debated = true;
        item.juryLabel = verdict.label;
        item.juryConfidence = verdict.confidence;
        item.juryCorrect = verdict.label === expected;
        item.juryStrategy = verdict.judgeStrategy;
        item.juryCostUsd = debateCost(verdict.totalCostUsd, primaryCostUsd);
        item.juryDurationMs = verdict.totalDurationMs;
        item.juryDegraded = verdict.debateDegraded;
        item.unpricedCalls = verdict.debateTranscript?.unpricedCalls ?? 0;
      }
      return item;
    });

    return new EvaluationReport(items, bandUpper);
  }
}
