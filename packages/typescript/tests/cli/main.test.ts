import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { main, resolveCalibrationLabels } from "../../src/cli/main.ts";

function tempDir(): string {
  return mkdtempSync(join(tmpdir(), "llm-jury-ts-"));
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

  const rc = await main([
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
  ]);

  assert.equal(rc, 0);
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
