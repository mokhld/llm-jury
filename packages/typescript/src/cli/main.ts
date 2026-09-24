#!/usr/bin/env node

import { readFileSync, realpathSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { LIBRARY_VERSION } from "../_version.ts";
import { ThresholdCalibrator } from "../calibration/optimizer.ts";
import { FunctionClassifier } from "../classifiers/functionAdapter.ts";
import { HuggingFaceClassifier } from "../classifiers/huggingFaceAdapter.ts";
import { LLMClassifier } from "../classifiers/llmClassifier.ts";
import { DebateConfig, DebateMode } from "../debate/engine.ts";
import { DEFAULT_MODEL } from "../defaults.ts";
import type { Verdict } from "../judges/base.ts";
import { BayesianJudge } from "../judges/bayesian.ts";
import { LLMJudge } from "../judges/llmJudge.ts";
import { MajorityVoteJudge } from "../judges/majorityVote.ts";
import { WeightedVoteJudge } from "../judges/weightedVote.ts";
import { Jury } from "../jury/core.ts";
import type { Persona } from "../personas/base.ts";
import { PersonaRegistry } from "../personas/registry.ts";

/**
 * Bad command-line usage: an unknown or malformed option, or input the command
 * cannot use. `runCli` prints the message to stderr and exits with code 2, the
 * same code the Python CLI uses for usage errors.
 */
export class CliUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CliUsageError";
  }
}

// ---------------------------------------------------------------------------
// Option parsing
// ---------------------------------------------------------------------------

const COMMANDS = ["classify", "calibrate"] as const;
type Command = (typeof COMMANDS)[number];

const COMMON_VALUE_OPTIONS = [
  "--classifier",
  "--personas",
  "--labels",
  "--judge",
  "--judge-model",
  "--persona-model",
  "--debate-mode",
  "--max-rounds",
  "--max-debate-cost",
  "--debate-concurrency",
];
const COMMON_FLAG_OPTIONS = ["--hide-primary-result", "--hide-confidence"];
const COMMAND_VALUE_OPTIONS: Record<Command, string[]> = {
  classify: ["--input", "--output", "--threshold", "--concurrency"],
  calibrate: ["--input", "--initial-threshold", "--error-cost", "--escalation-cost"],
};

type ParsedOptions = {
  values: Map<string, string>;
  flags: Set<string>;
};

/**
 * Parse `--name value`, `--name=value` and boolean flags for a command. Unknown
 * options, stray arguments and options missing their value are usage errors.
 */
function parseOptions(command: Command, args: string[]): ParsedOptions {
  const valueOptions = new Set([...COMMON_VALUE_OPTIONS, ...COMMAND_VALUE_OPTIONS[command]]);
  const flagOptions = new Set(COMMON_FLAG_OPTIONS);
  const values = new Map<string, string>();
  const flags = new Set<string>();

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i]!;
    const eq = arg.startsWith("--") ? arg.indexOf("=") : -1;
    const name = eq === -1 ? arg : arg.slice(0, eq);

    if (flagOptions.has(name)) {
      if (eq !== -1) {
        throw new CliUsageError(`Option '${name}' does not take a value.`);
      }
      flags.add(name);
      continue;
    }
    if (!valueOptions.has(name)) {
      throw new CliUsageError(
        arg.startsWith("-")
          ? `No such option for '${command}': ${name}`
          : `Got unexpected extra argument (${arg})`,
      );
    }

    let value: string | undefined;
    if (eq !== -1) {
      value = arg.slice(eq + 1);
    } else {
      value = args[i + 1];
      i += 1;
    }
    if (value === undefined) {
      throw new CliUsageError(`Option '${name}' requires an argument.`);
    }
    values.set(name, value);
  }

  return { values, flags };
}

type NumberRule = { integer?: boolean; min?: number; max?: number; expected: string };

const THRESHOLD_RULE: NumberRule = { min: 0, max: 1, expected: "a number between 0 and 1" };
const COUNT_RULE: NumberRule = { integer: true, min: 1, expected: "an integer >= 1" };
const COST_RULE: NumberRule = { min: 0, expected: "a finite number >= 0" };

// Plain decimal notation only: `Number()` alone would also accept "", "0x10"
// and "Infinity".
const DECIMAL_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
const INTEGER_PATTERN = /^[+-]?\d+$/;

function parseNumber(name: string, raw: string, rule: NumberRule): number {
  const text = raw.trim();
  const pattern = rule.integer ? INTEGER_PATTERN : DECIMAL_PATTERN;
  const value = pattern.test(text) ? Number(text) : Number.NaN;
  if (
    !Number.isFinite(value) ||
    (rule.min !== undefined && value < rule.min) ||
    (rule.max !== undefined && value > rule.max)
  ) {
    throw new CliUsageError(`Invalid value for '${name}': must be ${rule.expected}, got '${raw}'.`);
  }
  return value;
}

function numberOption(options: ParsedOptions, name: string, fallback: number, rule: NumberRule): number {
  const raw = options.values.get(name);
  return raw === undefined ? fallback : parseNumber(name, raw, rule);
}

function optionalNumberOption(options: ParsedOptions, name: string, rule: NumberRule): number | undefined {
  const raw = options.values.get(name);
  return raw === undefined ? undefined : parseNumber(name, raw, rule);
}

// ---------------------------------------------------------------------------
// JSONL input and output
// ---------------------------------------------------------------------------

function readJsonl(path: string): Array<Record<string, unknown>> {
  const lines = readFileSync(path, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.map((line) => JSON.parse(line) as Record<string, unknown>);
}

function writeJsonl(path: string, rows: unknown[]): void {
  writeFileSync(path, rows.map((row) => `${JSON.stringify(row)}\n`).join(""), "utf8");
}

// Keys whose values are data (classifier output, persona names) rather than
// library fields. Their contents are copied without renaming inner keys.
const DATA_KEYS = new Set(["rawOutput", "personaBiases"]);

function snakeCaseKey(key: string): string {
  if (!/^[a-z][A-Za-z0-9]*$/.test(key)) {
    return key;
  }
  return key.replace(/[A-Z]/g, (char) => `_${char.toLowerCase()}`);
}

/** Recursively rename camelCase identifier keys to snake_case. */
export function toSnakeCaseObject(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => toSnakeCaseObject(item));
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      out[snakeCaseKey(key)] = DATA_KEYS.has(key) ? entry : toSnakeCaseObject(entry);
    }
    return out;
  }
  return value;
}

/**
 * Output row for a verdict: the verdict's own JSON form (`toJSON`) with
 * snake_case keys, the same shape as the Python CLI's `Verdict.to_dict()` rows.
 * Optional fields that are unset are written as null.
 */
export function verdictToRow(verdict: Verdict): Record<string, unknown> {
  const json = JSON.stringify(verdict, (_key, value: unknown) => (value === undefined ? null : value));
  return toSnakeCaseObject(JSON.parse(json)) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

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

/** Labels passed with --labels, or null when the flag is absent or empty. */
function explicitLabels(raw: string | null): string[] | null {
  const labels = parseLabels(raw, []);
  return labels.length > 0 ? labels : null;
}

export function resolveCalibrationLabels(rawLabels: string | null, expectedLabels: string[]): string[] {
  return parseLabels(rawLabels, Array.from(new Set(expectedLabels)));
}

function selectPersonas(name: string) {
  switch (name.trim().toLowerCase()) {
    case "content_moderation":
      return PersonaRegistry.contentModeration();
    case "legal_compliance":
      return PersonaRegistry.legalCompliance();
    case "medical_triage":
      return PersonaRegistry.medicalTriage();
    case "financial_compliance":
      return PersonaRegistry.financialCompliance();
    default:
      throw new CliUsageError(`Unsupported personas set: ${name}`);
  }
}

function applyPersonaModel(personas: Persona[], model: string | null): Persona[] {
  if (!model) {
    return personas;
  }
  return personas.map((persona) => ({ ...persona, model }));
}

function selectJudge(name: string, model: string | null) {
  switch (name.trim().toLowerCase()) {
    case "llm":
      return new LLMJudge({ model: model ?? DEFAULT_MODEL });
    case "majority":
      return new MajorityVoteJudge();
    case "weighted":
      return new WeightedVoteJudge();
    case "bayesian":
      return new BayesianJudge();
    default:
      throw new CliUsageError(`Unsupported judge strategy: ${name}`);
  }
}

function buildDebateConfig(options: ParsedOptions): DebateConfig {
  const rawMode = options.values.get("--debate-mode") ?? DebateMode.INDEPENDENT;
  const mode = rawMode.trim().toLowerCase();
  const modeValues: string[] = Object.values(DebateMode);
  if (!modeValues.includes(mode)) {
    throw new CliUsageError(
      `Invalid value for '--debate-mode': unsupported debate mode '${rawMode}'. ` +
        `Use one of: ${modeValues.join(", ")}.`,
    );
  }

  return new DebateConfig({
    mode: mode as DebateMode,
    maxRounds: numberOption(options, "--max-rounds", 1, COUNT_RULE),
    includePrimaryResult: !options.flags.has("--hide-primary-result"),
    includeConfidence: !options.flags.has("--hide-confidence"),
  });
}

/** A stored prediction confidence in [0, 1], or null if it is missing or invalid. */
function predictionConfidence(value: unknown): number | null {
  let number: number;
  if (typeof value === "number") {
    number = value;
  } else if (typeof value === "string" && DECIMAL_PATTERN.test(value.trim())) {
    number = Number(value.trim());
  } else {
    return null;
  }
  return Number.isFinite(number) && number >= 0 && number <= 1 ? number : null;
}

function formatRowNumbers(rowNumbers: number[], limit = 5): string {
  const shown = rowNumbers.slice(0, limit).join(", ");
  return rowNumbers.length > limit ? `${shown} (and ${rowNumbers.length - limit} more)` : shown;
}

/**
 * Map each row's text to its stored `[predicted_label, predicted_confidence]`.
 * The `function` classifier replays predictions stored in the input file. It
 * never reads the ground-truth `label` field, so calibration cannot score the
 * ground truth against itself. Rows are numbered from 1 in error messages.
 */
function functionPredictions(rows: Array<Record<string, unknown>>): Map<string, [string, number]> {
  const predictions = new Map<string, [string, number]>();
  const invalidRows: number[] = [];
  const conflictingRows: number[] = [];

  rows.forEach((row, idx) => {
    const rowNumber = idx + 1;
    const label = row.predicted_label;
    const confidence = predictionConfidence(row.predicted_confidence);
    if (row.text == null || label == null || String(label).trim() === "" || confidence === null) {
      invalidRows.push(rowNumber);
      return;
    }
    const key = String(row.text);
    const prediction: [string, number] = [String(label), confidence];
    const existing = predictions.get(key);
    if (!existing) {
      predictions.set(key, prediction);
    } else if (existing[0] !== prediction[0] || existing[1] !== prediction[1]) {
      conflictingRows.push(rowNumber);
    }
  });

  if (invalidRows.length > 0) {
    throw new CliUsageError(
      "The 'function' classifier replays predictions stored in the input, so every row needs " +
        "'text', 'predicted_label' and 'predicted_confidence' (a number between 0 and 1). " +
        "The ground-truth 'label' field is never used as a prediction. Missing or invalid in " +
        `${invalidRows.length} row(s): ${formatRowNumbers(invalidRows)}.`,
    );
  }
  if (conflictingRows.length > 0) {
    throw new CliUsageError(
      "The 'function' classifier looks up predictions by text, but " +
        `${conflictingRows.length} row(s) repeat an earlier text with a different prediction: ` +
        `${formatRowNumbers(conflictingRows)}.`,
    );
  }
  return predictions;
}

/**
 * Build the primary classifier for a spec. `labels` is the label set for the
 * run. `labelsFlag` is what the user passed with --labels (null when absent);
 * the `huggingface:` spec uses it and otherwise takes label names from the
 * model's scores.
 */
export function buildClassifier(
  classifierSpec: string,
  labels: string[],
  rows: Array<Record<string, unknown>>,
  labelsFlag: string[] | null = null,
): { classifier: FunctionClassifier | LLMClassifier | HuggingFaceClassifier; isMockClassifier: boolean } {
  const spec = classifierSpec.trim();

  if (spec === "function") {
    const predictions = functionPredictions(rows);
    const lookup = (text: string): [string, number] => {
      const prediction = predictions.get(text);
      if (!prediction) {
        throw new Error(`No stored prediction for text: ${text}`);
      }
      return prediction;
    };
    return {
      classifier: new FunctionClassifier(lookup, labels),
      isMockClassifier: true,
    };
  }

  if (spec.startsWith("llm:")) {
    const model = spec.slice("llm:".length).trim();
    if (!model) {
      throw new CliUsageError("classifier spec 'llm:' requires a model name");
    }
    return {
      classifier: new LLMClassifier({ model, labels }),
      isMockClassifier: false,
    };
  }

  if (spec.startsWith("huggingface:")) {
    const modelName = spec.slice("huggingface:".length).trim();
    if (!modelName) {
      throw new CliUsageError("classifier spec 'huggingface:' requires a model name");
    }
    return {
      classifier: new HuggingFaceClassifier({ modelName, labels: labelsFlag ?? undefined }),
      isMockClassifier: false,
    };
  }

  throw new CliUsageError("Unsupported classifier spec. Use one of: function, llm:<model>, huggingface:<model>");
}

function usageText(): string {
  return [
    "Usage: llm-jury <command> [options]",
    "",
    "Commands:",
    "  classify   Classify JSONL inputs and write verdicts JSONL",
    "  calibrate  Calibrate threshold from labeled JSONL",
    "",
    "classify options:",
    "  --input <path>             Input JSONL file (required)",
    "  --output <path>            Output JSONL file (required)",
    "  --threshold <0-1>          Confidence threshold for escalation (default 0.7)",
    "  --concurrency <n>          Batch concurrency (default 10)",
    "",
    "calibrate options:",
    "  --input <path>             Input JSONL file with a ground-truth 'label' field (required)",
    "  --initial-threshold <0-1>  Starting threshold (default 0.7)",
    "  --error-cost <usd>         Cost per classification error (default 10)",
    "  --escalation-cost <usd>    Cost per escalation (default 0.05)",
    "",
    "Common options:",
    "  --classifier function|llm:<model>|huggingface:<model>  (default function)",
    "  --personas content_moderation|legal_compliance|medical_triage|financial_compliance",
    "  --labels safe,unsafe",
    "  --judge llm|majority|weighted|bayesian  (default llm)",
    "  --judge-model <model>",
    "  --persona-model <model>",
    "  --debate-mode independent|sequential|deliberation|adversarial  (default independent)",
    "  --max-rounds <n>           Max deliberation rounds (default 1)",
    "  --max-debate-cost <usd>    Max debate cost per item",
    "  --debate-concurrency <n>   Persona calls in flight per debate (default 5)",
    "  --hide-primary-result      Hide the primary result from personas",
    "  --hide-confidence          Hide the primary confidence from personas",
    "  --help, --version",
    "",
    "The 'function' classifier replays predictions stored in the input: every row needs",
    "'text', 'predicted_label' and 'predicted_confidence'.",
    "",
    "Examples:",
    "  llm-jury classify --input input.jsonl --output verdicts.jsonl --classifier function --judge majority",
    "  llm-jury calibrate --input calibration.jsonl --classifier function --judge majority",
  ].join("\n");
}

/**
 * Run a CLI command. Returns the exit code for completed runs and throws
 * `CliUsageError` for bad usage (`runCli` turns that into exit code 2).
 */
export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  if (argv.length === 0 || argv.includes("--help") || argv.includes("-h")) {
    process.stdout.write(`${usageText()}\n`);
    return 0;
  }

  if (argv.includes("--version") || argv.includes("-v")) {
    process.stdout.write(`${LIBRARY_VERSION}\n`);
    return 0;
  }

  const command = argv[0]!;
  if (!(COMMANDS as readonly string[]).includes(command)) {
    throw new CliUsageError("Supported commands: classify, calibrate");
  }
  const options = parseOptions(command as Command, argv.slice(1));
  const option = (name: string): string | null => options.values.get(name) ?? null;

  const classifierSpec = option("--classifier") ?? "function";
  const personasKey = option("--personas") ?? "content_moderation";
  const judgeKey = option("--judge") ?? "llm";
  const judgeModel = option("--judge-model") ?? DEFAULT_MODEL;
  const personaModel = option("--persona-model");
  const rawLabels = option("--labels");
  const labels = parseLabels(rawLabels, ["safe", "unsafe"]);
  const labelsFlag = explicitLabels(rawLabels);
  const debateConfig = buildDebateConfig(options);
  const debateConcurrency = numberOption(options, "--debate-concurrency", 5, COUNT_RULE);
  const maxDebateCostUsd = optionalNumberOption(options, "--max-debate-cost", COST_RULE);

  if (command === "classify") {
    const input = option("--input");
    const output = option("--output");
    if (!input || !output) {
      throw new CliUsageError("--input and --output are required");
    }
    const threshold = numberOption(options, "--threshold", 0.7, THRESHOLD_RULE);
    const concurrency = numberOption(options, "--concurrency", 10, COUNT_RULE);

    const rows = readJsonl(input);
    const texts = rows.map((row, idx) => String(row.text ?? `row-${idx}`));
    const { classifier, isMockClassifier } = buildClassifier(classifierSpec, labels, rows, labelsFlag);

    const jury = new Jury({
      classifier,
      personas: applyPersonaModel(selectPersonas(personasKey), personaModel),
      confidenceThreshold: threshold,
      judge: selectJudge(judgeKey, judgeModel),
      debateConfig,
      debateConcurrency,
      maxDebateCostUsd,
    });

    const results = await jury.classifyBatch(texts, isMockClassifier ? 1 : concurrency, true);
    let failures = 0;
    const outputRows = results.map((result, idx) => {
      if (result instanceof Error) {
        failures += 1;
        return { text: texts[idx], error: `${result.name}: ${result.message}` };
      }
      return verdictToRow(result);
    });
    writeJsonl(output, outputRows);
    process.stdout.write(`Wrote ${outputRows.length} verdict(s) to ${output}\n`);
    if (failures > 0) {
      process.stderr.write(
        `Warning: ${failures} of ${outputRows.length} row(s) failed; ` +
          "failed rows contain an 'error' field instead of a verdict.\n",
      );
      if (failures === outputRows.length) {
        return 1;
      }
    }
    return 0;
  }

  // calibrate
  const input = option("--input");
  if (!input) {
    throw new CliUsageError("--input is required");
  }
  const errorCost = numberOption(options, "--error-cost", 10, COST_RULE);
  const escalationCost = numberOption(options, "--escalation-cost", 0.05, COST_RULE);
  const initialThreshold = numberOption(options, "--initial-threshold", 0.7, THRESHOLD_RULE);

  const rows = readJsonl(input);
  if (rows.length === 0) {
    throw new CliUsageError("Input JSONL is empty.");
  }

  const missingLabels = rows.filter((row) => row.label == null).length;
  if (missingLabels > 0) {
    throw new CliUsageError(
      `Calibration input requires a ground-truth 'label' field on every row. Missing labels in ${missingLabels} row(s).`,
    );
  }

  const texts = rows.map((row, idx) => String(row.text ?? `row-${idx}`));
  const expectedLabels = rows.map((row) => String(row.label));
  const inferenceLabels = resolveCalibrationLabels(rawLabels, expectedLabels);
  const { classifier } = buildClassifier(classifierSpec, inferenceLabels, rows, labelsFlag);

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

/**
 * CLI entry point: runs `main` and maps failures to exit codes. Usage errors
 * print `Error: <message>` to stderr and return 2; unexpected errors print the
 * error and return 1.
 */
export async function runCli(argv: string[] = process.argv.slice(2)): Promise<number> {
  try {
    return await main(argv);
  } catch (err) {
    if (err instanceof CliUsageError) {
      process.stderr.write(`Error: ${err.message}\nTry 'llm-jury --help' for help.\n`);
      return 2;
    }
    console.error(err);
    return 1;
  }
}

/**
 * True when this file is the script node was started with, e.g.
 * `node dist/cli/main.js`. Both sides are resolved with realpath so symlinks,
 * relative paths and paths with spaces compare equal.
 */
function isEntryPoint(): boolean {
  const entry = process.argv[1];
  if (!entry) {
    return false;
  }
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isEntryPoint()) {
  void runCli().then((code) => {
    process.exitCode = code;
  });
}
