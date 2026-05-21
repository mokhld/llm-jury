/**
 * Threshold calibration using labelled data.
 *
 * Uses FunctionClassifier to demonstrate the calibration workflow
 * without requiring API calls. In production, swap in your real
 * classifier (LLMClassifier, HuggingFaceClassifier, etc.).
 *
 * Run with:
 *   node --experimental-strip-types examples/typescript/threshold_calibration.ts
 */
import {
  FunctionClassifier,
  Jury,
  ThresholdCalibrator,
} from "@llm-jury/core";

async function main(): Promise<void> {
  // Simulated classifier outputs for calibration data.
  const predictions: Record<string, [string, number]> = {
    a: ["safe", 0.9],
    b: ["unsafe", 0.4],
    c: ["safe", 0.65],
    d: ["unsafe", 0.55],
  };

  const classifier = new FunctionClassifier(
    (text) => predictions[text]!,
    ["safe", "unsafe"],
  );

  const jury = new Jury({ classifier, personas: [], confidenceThreshold: 0.7 });
  const calibrator = new ThresholdCalibrator(jury);

  const best = await calibrator.calibrate({
    texts: ["a", "b", "c", "d"],
    labels: ["safe", "unsafe", "safe", "unsafe"],
    thresholds: [0.5, 0.6, 0.7, 0.8],
  });

  console.log(`Best threshold: ${best}`);
  console.log();

  const report = calibrator.calibrationReport();
  for (const row of report.rows) {
    console.log(
      `  threshold=${row.threshold.toFixed(2)}  ` +
        `accuracy=${row.accuracy.toFixed(2)}  ` +
        `escalation_rate=${row.escalationRate.toFixed(2)}  ` +
        `cost=${row.totalCost.toFixed(2)}`,
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
