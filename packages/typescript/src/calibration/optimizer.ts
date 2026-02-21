import type { Jury } from "../jury/core.ts";

export type CalibrationOptions = {
  texts: string[];
  labels: string[];
  errorCost?: number;
  escalationCost?: number;
  thresholds?: number[];
};

type CalibrationRow = {
  threshold: number;
  accuracy: number;
  escalationRate: number;
  totalCost: number;
};

export class ThresholdCalibrator {
  private jury: Jury;
  private rows: CalibrationRow[] = [];
  private bestThreshold: number | null = null;

  constructor(jury: Jury) {
    this.jury = jury;
  }

  async calibrate(options: CalibrationOptions): Promise<number> {
    const errorCost = options.errorCost ?? 10;
    const escalationCost = options.escalationCost ?? 0.05;
    const thresholds =
      options.thresholds ?? Array.from({ length: 10 }, (_v, idx) => Number((0.5 + idx * 0.05).toFixed(2)));

    if (options.texts.length !== options.labels.length) {
      throw new Error("texts and labels must have same length");
    }

    this.rows = [];
    let bestThreshold = thresholds[0] ?? 0.7;
    let bestCost = Number.POSITIVE_INFINITY;

    for (const threshold of thresholds) {
      let errors = 0;
      let escalations = 0;
      let correct = 0;

      for (let i = 0; i < options.texts.length; i += 1) {
        const text = options.texts[i]!;
        const expected = options.labels[i]!;
        const result = await this.jury.classifier.classify(text);

        if (result.confidence < threshold) {
          escalations += 1;
          correct += 1;
        } else if (result.label === expected) {
          correct += 1;
        } else {
          errors += 1;
        }
      }

      const total = Math.max(1, options.texts.length);
      const totalCost = errors * errorCost + escalations * escalationCost;
      const row: CalibrationRow = {
        threshold,
        accuracy: correct / total,
        escalationRate: escalations / total,
        totalCost,
      };
      this.rows.push(row);

      if (totalCost < bestCost) {
        bestCost = totalCost;
        bestThreshold = threshold;
      }
    }

    this.bestThreshold = bestThreshold;
    this.jury.threshold = bestThreshold;
    return bestThreshold;
  }

  calibrationReport(): { bestThreshold: number | null; rows: CalibrationRow[] } {
    return {
      bestThreshold: this.bestThreshold,
      rows: [...this.rows],
    };
  }
}
