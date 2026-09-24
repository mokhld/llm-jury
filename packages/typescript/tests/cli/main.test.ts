import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LIBRARY_VERSION } from "../../src/_version.ts";
import {
  buildClassifier,
  CliUsageError,
  main,
  resolveCalibrationLabels,
  runCli,
  verdictToRow,
} from "../../src/cli/main.ts";
import { Verdict } from "../../src/judges/base.ts";
import { Jury } from "../../src/jury/core.ts";

function tempDir(): string {
  return mkdtempSync(join(tmpdir(), "llm-jury-ts-"));
}

function writeRows(path: string, rows: Array<Record<string, unknown>>): void {
  writeFileSync(path, rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
}

async function captureOutput<T>(fn: () => Promise<T>): Promise<{ result: T; stdout: string; stderr: string }> {
  const stdout: string[] = [];
  const stderr: string[] = [];
  const originalOut = process.stdout.write.bind(process.stdout);
  const originalErr = process.stderr.write.bind(process.stderr);
  process.stdout.write = ((chunk: string | Uint8Array) => {
    stdout.push(String(chunk));
    return true;
  }) as typeof process.stdout.write;
  process.stderr.write = ((chunk: string | Uint8Array) => {
    stderr.push(String(chunk));
    return true;
  }) as typeof process.stderr.write;
  try {
    const result = await fn();
    return { result, stdout: stdout.join(""), stderr: stderr.join("") };
  } finally {
    process.stdout.write = originalOut;
    process.stderr.write = originalErr;
  }
}

const FUNCTION_ARGS = ["--classifier", "function", "--judge", "majority"];

function classifyArgs(dir: string, rows?: Array<Record<string, unknown>>): string[] {
  const inputPath = join(dir, "input.jsonl");
  writeRows(inputPath, rows ?? [{ text: "a", predicted_label: "safe", predicted_confidence: 0.9 }]);
  return ["classify", "--input", inputPath, "--output", join(dir, "output.jsonl"), ...FUNCTION_ARGS];
}

function calibrateArgs(dir: string): string[] {
  const inputPath = join(dir, "calib.jsonl");
  writeRows(inputPath, [{ text: "a", label: "safe", predicted_label: "safe", predicted_confidence: 0.9 }]);
  return ["calibrate", "--input", inputPath, ...FUNCTION_ARGS];
}

test("cli classify writes output", async () => {
  const dir = tempDir();
  const inputPath = join(dir, "input.jsonl");
  const outputPath = join(dir, "output.jsonl");

  writeFileSync(
    inputPath,
    [
      JSON.stringify({ text: "a", predicted_label: "safe", predicted_confidence: 0.95 }),
      JSON.stringify({ text: "b", predicted_label: "unsafe", predicted_confidence: 0.96 }),
    ].join("\n") + "\n",
    "utf8",
  );

  const { result: rc, stdout } = await captureOutput(() =>
    main([
      "classify",
      "--input",
      inputPath,
      "--output",
      outputPath,
      "--classifier",
      "function",
      "--personas",
      "content_moderation",
      "--judge",
      "majority",
      "--labels",
      "safe,unsafe",
      "--threshold",
      "0.7",
    ]),
  );

  assert.equal(rc, 0);
  assert.match(stdout, /Wrote 2 verdict\(s\)/);
  const lines = readFileSync(outputPath, "utf8").trim().split("\n");
  assert.equal(lines.length, 2);
  const first = JSON.parse(lines[0] ?? "{}");
  assert.ok(typeof first.label === "string");
  assert.ok(typeof first.was_escalated === "boolean");
});

test("cli calibrate returns success", async () => {
  const dir = tempDir();
  const inputPath = join(dir, "calib.jsonl");

  writeFileSync(
    inputPath,
    [
      JSON.stringify({ text: "t1", label: "safe", predicted_label: "safe", predicted_confidence: 0.9 }),
      JSON.stringify({ text: "t2", label: "unsafe", predicted_label: "unsafe", predicted_confidence: 0.4 }),
    ].join("\n") + "\n",
    "utf8",
  );

  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (() => true) as typeof process.stdout.write;
  let rc = 1;
  try {
    rc = await main([
      "calibrate",
      "--input",
      inputPath,
      "--classifier",
      "function",
      "--personas",
      "content_moderation",
      "--judge",
      "majority",
      "--labels",
      "safe,unsafe",
    ]);
  } finally {
    process.stdout.write = originalWrite;
  }

  assert.equal(rc, 0);
});

test("cli help returns success", async () => {
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (() => true) as typeof process.stdout.write;
  let rc = 1;
  try {
    rc = await main(["--help"]);
  } finally {
    process.stdout.write = originalWrite;
  }
  assert.equal(rc, 0);
});

test("cli calibrate requires ground-truth label", async () => {
  const dir = tempDir();
  const inputPath = join(dir, "calib-missing-label.jsonl");

  writeFileSync(
    inputPath,
    [
      JSON.stringify({ text: "t1", predicted_label: "safe", predicted_confidence: 0.9 }),
      JSON.stringify({ text: "t2", predicted_label: "unsafe", predicted_confidence: 0.4 }),
    ].join("\n") + "\n",
    "utf8",
  );

  await assert.rejects(
    main([
      "calibrate",
      "--input",
      inputPath,
      "--classifier",
      "function",
      "--personas",
      "content_moderation",
      "--judge",
      "majority",
      "--labels",
      "safe,unsafe",
    ]),
    /ground-truth 'label'/,
  );
});

test("calibration labels fallback to dataset labels when --labels is absent", () => {
  const labels = resolveCalibrationLabels(null, ["spam", "ham", "spam"]);
  assert.deepEqual(labels, ["spam", "ham"]);
});

// --- version ---------------------------------------------------------------

test("cli --version prints the library version", async () => {
  const { result, stdout } = await captureOutput(() => main(["--version"]));
  assert.equal(result, 0);
  assert.equal(stdout, `${LIBRARY_VERSION}\n`);
});

// --- function classifier spec ----------------------------------------------

test("calibrate rejects rows without stored predictions instead of using ground truth", async () => {
  // Rows with only text + ground-truth label used to be scored against
  // themselves: accuracy 1.0 at every threshold.
  const dir = tempDir();
  const inputPath = join(dir, "calib.jsonl");
  writeRows(inputPath, [
    { text: "t1", label: "safe" },
    { text: "t2", label: "unsafe" },
  ]);

  await assert.rejects(
    main(["calibrate", "--input", inputPath, ...FUNCTION_ARGS]),
    (err: unknown) =>
      err instanceof CliUsageError && /predicted_label/.test(err.message) && /2 row\(s\): 1, 2\./.test(err.message),
  );

  const { result, stderr } = await captureOutput(() => runCli(["calibrate", "--input", inputPath, ...FUNCTION_ARGS]));
  assert.equal(result, 2);
  assert.match(stderr, /^Error: The 'function' classifier/);
});

test("classify rejects rows without stored predictions and writes nothing", async () => {
  const dir = tempDir();
  const args = classifyArgs(dir, [
    { text: "a", predicted_label: "safe", predicted_confidence: 0.9 },
    { text: "b", label: "unsafe" },
  ]);
  const { result } = await captureOutput(() => runCli(args));
  assert.equal(result, 2);
  assert.equal(existsSync(join(dir, "output.jsonl")), false);
});

test("function spec reads predicted fields, never the ground-truth label", async () => {
  const { classifier, isMockClassifier } = buildClassifier(
    "function",
    ["safe", "unsafe"],
    [
      { text: "t1", label: "safe", predicted_label: "unsafe", predicted_confidence: 0.8 },
      { text: "t2", label: "unsafe", predicted_label: "unsafe", predicted_confidence: "0.6" },
    ],
  );
  assert.equal(isMockClassifier, true);
  const first = await classifier.classify("t1");
  const second = await classifier.classify("t2");
  assert.deepEqual([first.label, first.confidence], ["unsafe", 0.8]);
  assert.deepEqual([second.label, second.confidence], ["unsafe", 0.6]);
});

test("function spec error names at most five rows", () => {
  const rows = Array.from({ length: 8 }, (_, i) => ({ text: `t${i}`, label: "safe" }));
  assert.throws(() => buildClassifier("function", ["safe", "unsafe"], rows), /8 row\(s\): 1, 2, 3, 4, 5 \(and 3 more\)\./);
});

test("function spec rejects missing or invalid predicted values", () => {
  const badRows: Array<Record<string, unknown>> = [
    { predicted_label: "safe", predicted_confidence: 0.9 },
    { text: "x", predicted_label: null, predicted_confidence: 0.9 },
    { text: "x", predicted_label: "  ", predicted_confidence: 0.9 },
    { text: "x", predicted_label: "safe" },
    { text: "x", predicted_label: "safe", predicted_confidence: "high" },
    { text: "x", predicted_label: "safe", predicted_confidence: 1.5 },
    { text: "x", predicted_label: "safe", predicted_confidence: -0.1 },
    { text: "x", predicted_label: "safe", predicted_confidence: true },
    { text: "x", predicted_label: "safe", predicted_confidence: "" },
    { text: "x", predicted_label: "safe", predicted_confidence: [0.9] },
  ];
  for (const row of badRows) {
    assert.throws(
      () => buildClassifier("function", ["safe", "unsafe"], [row]),
      (err: unknown) => err instanceof CliUsageError && /1 row\(s\): 1\./.test(err.message),
      JSON.stringify(row),
    );
  }
});

test("function spec allows a repeated text only when its prediction matches", async () => {
  const row = { text: "dup", predicted_label: "safe", predicted_confidence: 0.9 };
  const { classifier } = buildClassifier("function", ["safe", "unsafe"], [row, { ...row }]);
  assert.equal((await classifier.classify("dup")).label, "safe");

  assert.throws(
    () =>
      buildClassifier(
        "function",
        ["safe", "unsafe"],
        [row, { text: "other", predicted_label: "safe", predicted_confidence: 0.9 }, { ...row, predicted_label: "unsafe" }],
      ),
    /repeat an earlier text with a different prediction: 3\./,
  );
});

// --- huggingface spec --------------------------------------------------------

test("huggingface spec passes --labels to the classifier", () => {
  const withFlag = buildClassifier("huggingface:some/model", ["safe", "unsafe"], [], ["safe", "unsafe"]);
  assert.equal(withFlag.isMockClassifier, false);
  assert.deepEqual(withFlag.classifier.labels, ["safe", "unsafe"]);

  const withoutFlag = buildClassifier("huggingface:some/model", ["safe", "unsafe"], []);
  assert.deepEqual(withoutFlag.classifier.labels, []);
});

// --- option validation -------------------------------------------------------

test("numeric options are validated with exit code 2", async () => {
  const classifyCases: Array<[string, string]> = [
    ["--concurrency", "ten"],
    ["--concurrency", "0"],
    ["--concurrency", "2.5"],
    ["--debate-concurrency", "five"],
    ["--debate-concurrency", "0"],
    ["--max-rounds", "0"],
    ["--threshold", "0,7"],
    ["--threshold", "1.5"],
    ["--threshold", "-0.1"],
    ["--threshold", ""],
    ["--threshold", "0x1"],
    ["--threshold", "Infinity"],
    ["--max-debate-cost", "-1"],
    ["--max-debate-cost", "1e400"],
  ];
  for (const [flag, value] of classifyCases) {
    const dir = tempDir();
    const args = [...classifyArgs(dir), flag, value];
    await assert.rejects(
      main(args),
      (err: unknown) => err instanceof CliUsageError && err.message.includes(`'${flag}'`),
      `${flag} ${value}`,
    );
    const { result, stderr } = await captureOutput(() => runCli(args));
    assert.equal(result, 2, `${flag} ${value}`);
    assert.match(stderr, new RegExp(`Invalid value for '${flag}'`));
    assert.equal(existsSync(join(dir, "output.jsonl")), false);
  }

  const calibrateCases: Array<[string, string]> = [
    ["--initial-threshold", "2"],
    ["--error-cost", "-1"],
    ["--escalation-cost", "abc"],
  ];
  for (const [flag, value] of calibrateCases) {
    const dir = tempDir();
    const { result, stderr } = await captureOutput(() => runCli([...calibrateArgs(dir), flag, value]));
    assert.equal(result, 2, `${flag} ${value}`);
    assert.match(stderr, new RegExp(`Invalid value for '${flag}'`));
  }
});

test("valid numeric options and --name=value syntax are accepted", async () => {
  const dir = tempDir();
  const args = [...classifyArgs(dir), "--threshold=0.5", "--concurrency", "3", "--max-debate-cost", "0.25"];
  const { result } = await captureOutput(() => main(args));
  assert.equal(result, 0);
});

test("unknown options, missing values and bad debate modes are usage errors", async () => {
  const dir = tempDir();
  const cases: Array<[string[], RegExp]> = [
    [["--treshold", "0.9"], /No such option for 'classify': --treshold/],
    [["--error-cost", "1"], /No such option for 'classify': --error-cost/],
    [["stray"], /unexpected extra argument \(stray\)/],
    [["--threshold"], /'--threshold' requires an argument/],
    [["--debate-mode", "chaos"], /independent, sequential, deliberation, adversarial/],
    [["--judge", "oracle"], /Unsupported judge strategy/],
  ];
  for (const [extra, pattern] of cases) {
    const { result, stderr } = await captureOutput(() => runCli([...classifyArgs(dir), ...extra]));
    assert.equal(result, 2, extra.join(" "));
    assert.match(stderr, pattern);
  }

  const { result, stderr } = await captureOutput(() => runCli(["frobnicate"]));
  assert.equal(result, 2);
  assert.match(stderr, /Supported commands: classify, calibrate/);
});

// --- output rows ---------------------------------------------------------------

// Top-level keys of the Python CLI's Verdict.to_dict() rows.
const PYTHON_VERDICT_KEYS = [
  "label",
  "confidence",
  "reasoning",
  "was_escalated",
  "primary_result",
  "debate_transcript",
  "judge_strategy",
  "total_duration_ms",
  "total_cost_usd",
  "persona_failures",
  "debate_degraded",
  "library_version",
  "created_at",
];

test("classify output rows have the Python Verdict.to_dict() keys", async () => {
  const dir = tempDir();
  const args = classifyArgs(dir);
  const { result } = await captureOutput(() => main(args));
  assert.equal(result, 0);
  const row = JSON.parse(readFileSync(join(dir, "output.jsonl"), "utf8").trim());
  for (const key of PYTHON_VERDICT_KEYS) {
    assert.ok(key in row, `missing ${key}`);
  }
  assert.equal(row.debate_degraded, false);
  assert.equal(row.library_version, LIBRARY_VERSION);
  assert.equal(row.debate_transcript, null);
  assert.equal(row.primary_result.label, "safe");
});

test("verdictToRow snake-cases library fields and keeps data keys as given", () => {
  const primaryResult = { label: "unsafe", confidence: 0.4, rawOutput: { topLabel: "unsafe", "Raw Score": 1 } };
  const transcript = {
    inputText: "hello",
    primaryResult,
    rounds: [
      [
        {
          personaName: "Policy Analyst",
          label: "unsafe",
          confidence: 0.9,
          reasoning: "r",
          keyFactors: ["k"],
          failed: false,
        },
        {
          personaName: "Free Speech Advocate",
          label: "",
          confidence: 0,
          reasoning: "call failed",
          keyFactors: [],
          failed: true,
        },
      ],
    ],
    summary: undefined,
    durationMs: 5,
    totalTokens: 10,
    totalCostUsd: 0.01,
    personaBiases: { "Policy Analyst": "strict", "Free Speech Advocate": "lenient" },
  };
  const verdict = new Verdict({
    label: "unsafe",
    confidence: 0.9,
    reasoning: "majority",
    wasEscalated: true,
    primaryResult,
    debateTranscript: transcript,
    judgeStrategy: "majority_vote",
    totalDurationMs: 7,
    totalCostUsd: 0.01,
    personaFailures: 1,
  });

  const row = verdictToRow(verdict);
  for (const key of PYTHON_VERDICT_KEYS) {
    assert.ok(key in row, `missing ${key}`);
  }
  assert.equal(row.debate_degraded, true);
  assert.equal(row.persona_failures, 1);

  const primary = row.primary_result as Record<string, unknown>;
  assert.deepEqual(primary.raw_output, { topLabel: "unsafe", "Raw Score": 1 });

  const debate = row.debate_transcript as Record<string, unknown>;
  assert.equal(debate.input_text, "hello");
  assert.equal(debate.total_cost_usd, 0.01);
  assert.equal(debate.summary, null);
  assert.deepEqual(debate.persona_biases, { "Policy Analyst": "strict", "Free Speech Advocate": "lenient" });
  const firstResponse = (debate.rounds as Array<Array<Record<string, unknown>>>)[0]![0]!;
  assert.equal(firstResponse.persona_name, "Policy Analyst");
  assert.deepEqual(firstResponse.key_factors, ["k"]);
});

test("classify writes error rows like the Python CLI and exits 1 only when every row fails", async () => {
  const originalClassify = Jury.prototype.classify;
  Jury.prototype.classify = async function (this: Jury, text: string) {
    if (text.startsWith("boom")) {
      throw new TypeError("row failed");
    }
    return originalClassify.call(this, text);
  };
  try {
    const dir = tempDir();
    const partial = classifyArgs(dir, [
      { text: "boom", predicted_label: "safe", predicted_confidence: 0.95 },
      { text: "fine", predicted_label: "safe", predicted_confidence: 0.95 },
    ]);
    const first = await captureOutput(() => runCli(partial));
    assert.equal(first.result, 0);
    assert.match(first.stderr, /Warning: 1 of 2 row\(s\) failed/);
    const lines = readFileSync(join(dir, "output.jsonl"), "utf8").trim().split("\n");
    assert.deepEqual(JSON.parse(lines[0]!), { text: "boom", error: "TypeError: row failed" });
    assert.equal(JSON.parse(lines[1]!).label, "safe");

    const allFailDir = tempDir();
    const allFail = classifyArgs(allFailDir, [
      { text: "boom1", predicted_label: "safe", predicted_confidence: 0.95 },
      { text: "boom2", predicted_label: "safe", predicted_confidence: 0.95 },
    ]);
    const second = await captureOutput(() => runCli(allFail));
    assert.equal(second.result, 1);
  } finally {
    Jury.prototype.classify = originalClassify;
  }
});

test("classify on an empty input writes an empty file", async () => {
  const dir = tempDir();
  const inputPath = join(dir, "input.jsonl");
  const outputPath = join(dir, "output.jsonl");
  writeFileSync(inputPath, "", "utf8");
  const { result } = await captureOutput(() =>
    main(["classify", "--input", inputPath, "--output", outputPath, ...FUNCTION_ARGS]),
  );
  assert.equal(result, 0);
  assert.equal(readFileSync(outputPath, "utf8"), "");
});
