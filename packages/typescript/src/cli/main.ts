#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";

import { ThresholdCalibrator } from "../calibration/optimizer.ts";
import { FunctionClassifier } from "../classifiers/functionAdapter.ts";
import { HuggingFaceClassifier } from "../classifiers/huggingFaceAdapter.ts";
import { LLMClassifier } from "../classifiers/llmClassifier.ts";
import { DebateConfig, DebateMode } from "../debate/engine.ts";
import { BayesianJudge } from "../judges/bayesian.ts";
import { LLMJudge } from "../judges/llmJudge.ts";
import { MajorityVoteJudge } from "../judges/majorityVote.ts";
import { WeightedVoteJudge } from "../judges/weightedVote.ts";
import { Jury } from "../jury/core.ts";
import type { Persona } from "../personas/base.ts";
import { PersonaRegistry } from "../personas/registry.ts";

function parseArg(argv: string[], name: string): string | null {
  const idx = argv.indexOf(name);
  if (idx === -1 || idx + 1 >= argv.length) {
    return null;
  }
  return argv[idx + 1] ?? null;
}

function parseBoolFlag(argv: string[], name: string): boolean {
  return argv.includes(name);
}

function readJsonl(path: string): Array<Record<string, unknown>> {
  const lines = readFileSync(path, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.map((line) => JSON.parse(line) as Record<string, unknown>);
}

function writeJsonl(path: string, rows: unknown[]): void {
  writeFileSync(path, rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
}

function toSnakeCaseObject(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => toSnakeCaseObject(item));
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      const snake = key.replace(/[A-Z]/g, (char) => `_${char.toLowerCase()}`);
      out[snake] = toSnakeCaseObject(entry);
    }
    return out;
  }
  return value;
}

export function parseLabels(raw: string | null, fallback: string[] = ["safe", "unsafe"]): string[] {
  if (!raw) {
    return fallback;
  }
  const labels = raw
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean);
  return labels.length > 0 ? labels : fallback;
}

export function resolveCalibrationLabels(rawLabels: string | null, expectedLabels: string[]): string[] {
  return parseLabels(rawLabels, Array.from(new Set(expectedLabels)));
}

function selectPersonas(name: string) {
  switch (name.toLowerCase()) {
    case "content_moderation":
      return PersonaRegistry.contentModeration();
    case "legal_compliance":
      return PersonaRegistry.legalCompliance();
    case "medical_triage":
      return PersonaRegistry.medicalTriage();
    case "financial_compliance":
      return PersonaRegistry.financialCompliance();
    default:
      throw new Error(`Unsupported personas set: ${name}`);
  }
}

function applyPersonaModel(personas: Persona[], model: string | null): Persona[] {
  if (!model) {
    return personas;
  }
  return personas.map((persona) => ({ ...persona, model }));
}

function selectJudge(name: string, model: string | null) {
  switch (name.toLowerCase()) {
    case "llm":
      return new LLMJudge({ model: model ?? "gpt-5-mini" });
    case "majority":
      return new MajorityVoteJudge();
    case "weighted":
      return new WeightedVoteJudge();
    case "bayesian":
      return new BayesianJudge();
    default:
      throw new Error(`Unsupported judge strategy: ${name}`);
  }
}

function buildDebateConfig(argv: string[]): DebateConfig {
  const rawMode = parseArg(argv, "--debate-mode") ?? DebateMode.INDEPENDENT;
  const modeValues = Object.values(DebateMode);
  if (!modeValues.includes(rawMode as (typeof modeValues)[number])) {
    throw new Error(`Unsupported debate mode: ${rawMode}`);
  }

  const maxRounds = Number(parseArg(argv, "--max-rounds") ?? "1");
  return new DebateConfig({
    mode: rawMode as (typeof modeValues)[number],
    maxRounds,
    includePrimaryResult: !parseBoolFlag(argv, "--hide-primary-result"),
    includeConfidence: !parseBoolFlag(argv, "--hide-confidence"),
  });
}

function buildClassifier(
  classifierSpec: string,
  labels: string[],
  rows: Array<Record<string, unknown>>,
): { classifier: FunctionClassifier | LLMClassifier | HuggingFaceClassifier; isMockClassifier: boolean } {
  if (classifierSpec === "function") {
    const predictionMap = new Map<string, [string, number]>();
    rows.forEach((row, idx) => {
      const text = String(row.text ?? `row-${idx}`);
      const predictedLabel = String(row.predicted_label ?? row.label ?? labels[0] ?? "unknown");
      const predictedConfidence = Number(row.predicted_confidence ?? 0.95);
      predictionMap.set(text, [predictedLabel, predictedConfidence]);
    });

    return {
      classifier: new FunctionClassifier((text) => predictionMap.get(text) ?? [labels[0] ?? "unknown", 0.95], labels),
      isMockClassifier: true,
    };
  }

  if (classifierSpec.startsWith("llm:")) {
    const model = classifierSpec.slice("llm:".length).trim();
    if (!model) {
      throw new Error("classifier spec 'llm:' requires a model name");
    }
    return {
      classifier: new LLMClassifier({ model, labels }),
      isMockClassifier: false,
    };
  }

  if (classifierSpec.startsWith("huggingface:")) {
    const modelName = classifierSpec.slice("huggingface:".length).trim();
    if (!modelName) {
      throw new Error("classifier spec 'huggingface:' requires a model name");
    }
    return {
      classifier: new HuggingFaceClassifier({ modelName }),
      isMockClassifier: false,
    };
  }

  throw new Error("Unsupported classifier spec. Use: function, llm:<model>, huggingface:<model>");
}

function usageText(): string {
  return [
    "Usage: llm-jury <command> [options]",
    "",
    "Commands:",
    "  classify   Classify JSONL inputs and write verdicts JSONL",
    "  calibrate  Calibrate threshold from labeled JSONL",
    "",
    "Common options:",
    "  --classifier function|llm:<model>|huggingface:<model>",
    "  --personas content_moderation|legal_compliance|medical_triage|financial_compliance",
    "  --judge llm|majority|weighted|bayesian",
    "  --labels safe,unsafe",
    "  --debate-mode independent|sequential|deliberation|adversarial",
    "",
    "Examples:",
    "  llm-jury classify --input input.jsonl --output verdicts.jsonl --classifier function --judge majority",
    "  llm-jury calibrate --input calibration.jsonl --classifier function --judge majority",
  ].join("\n");
}

export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  if (argv.length === 0 || argv.includes("--help") || argv.includes("-h")) {
    process.stdout.write(`${usageText()}\n`);
    return 0;
  }

  if (argv.includes("--version") || argv.includes("-v")) {
    process.stdout.write("0.1.0\n");
    return 0;
  }

  const command = argv[0];
  const classifierSpec = parseArg(argv, "--classifier") ?? "function";
  const personasKey = parseArg(argv, "--personas") ?? "content_moderation";
  const judgeKey = parseArg(argv, "--judge") ?? "llm";
  const judgeModel = parseArg(argv, "--judge-model") ?? "gpt-5-mini";
  const personaModel = parseArg(argv, "--persona-model") ?? "gpt-5-mini";
  const rawLabels = parseArg(argv, "--labels");
  const labels = parseLabels(rawLabels, ["safe", "unsafe"]);
  const debateConfig = buildDebateConfig(argv);
  const debateConcurrency = Number(parseArg(argv, "--debate-concurrency") ?? "5");
  const maxDebateCostRaw = parseArg(argv, "--max-debate-cost");
  const maxDebateCostUsd = maxDebateCostRaw == null ? undefined : Number(maxDebateCostRaw);

  if (command === "classify") {
    const input = parseArg(argv, "--input");
    const output = parseArg(argv, "--output");
    const thresholdValue = parseArg(argv, "--threshold");
    const concurrency = Number(parseArg(argv, "--concurrency") ?? "10");

    if (!input || !output) {
      throw new Error("--input and --output are required");
    }

    const threshold = thresholdValue ? Number(thresholdValue) : 0.7;
    const rows = readJsonl(input);
    const texts = rows.map((row, idx) => String(row.text ?? `row-${idx}`));
    const { classifier, isMockClassifier } = buildClassifier(classifierSpec, labels, rows);

    const jury = new Jury({
      classifier,
      personas: applyPersonaModel(selectPersonas(personasKey), personaModel),
      confidenceThreshold: threshold,
      judge: selectJudge(judgeKey, judgeModel),
      debateConfig,
      debateConcurrency,
      maxDebateCostUsd,
    });

    const verdicts = await jury.classifyBatch(texts, isMockClassifier ? 1 : concurrency);
    writeJsonl(output, verdicts.map((verdict) => toSnakeCaseObject(verdict)));
    return 0;
  }

  if (command === "calibrate") {
    const input = parseArg(argv, "--input");
    const errorCost = Number(parseArg(argv, "--error-cost") ?? "10");
    const escalationCost = Number(parseArg(argv, "--escalation-cost") ?? "0.05");
    const initialThreshold = Number(parseArg(argv, "--initial-threshold") ?? "0.7");

    if (!input) {
      throw new Error("--input is required");
    }

    const rows = readJsonl(input);
    if (rows.length === 0) {
      throw new Error("input jsonl is empty");
    }

    const missingLabels = rows.filter((row) => row.label == null).length;
    if (missingLabels > 0) {
      throw new Error(
        `Calibration input requires a ground-truth 'label' field on every row. Missing labels in ${missingLabels} row(s).`,
      );
    }

    const texts = rows.map((row, idx) => String(row.text ?? `row-${idx}`));
    const expectedLabels = rows.map((row) => String(row.label));
    const inferenceLabels = resolveCalibrationLabels(rawLabels, expectedLabels);
    const { classifier } = buildClassifier(classifierSpec, inferenceLabels, rows);

    const jury = new Jury({
      classifier,
      personas: applyPersonaModel(selectPersonas(personasKey), personaModel),
      confidenceThreshold: initialThreshold,
      judge: selectJudge(judgeKey, judgeModel),
      debateConfig,
      debateConcurrency,
      maxDebateCostUsd,
    });

    const calibrator = new ThresholdCalibrator(jury);
    const bestThreshold = await calibrator.calibrate({
      texts,
      labels: expectedLabels,
      errorCost,
      escalationCost,
    });
    const report = calibrator.calibrationReport();
    report.bestThreshold = bestThreshold;
    process.stdout.write(`${JSON.stringify(toSnakeCaseObject(report))}\n`);
    return 0;
  }

  throw new Error("Supported commands: classify, calibrate");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().then(
    (code) => {
      process.exitCode = code;
    },
    (err) => {
      console.error(err);
      process.exitCode = 1;
    },
  );
}
