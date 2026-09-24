import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { LIBRARY_VERSION } from "../../src/_version.ts";

const PACKAGE_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const CLI_DIR = join(PACKAGE_DIR, "src", "cli");

// npm installs the bin as a symlink (node_modules/.bin/llm-jury), so run the CLI
// the same way: through a symlink, from a directory whose path has spaces.
function runThroughSymlink(target: string, args: string[]) {
  const dir = mkdtempSync(join(tmpdir(), "llm jury bin "));
  const link = join(dir, "llm-jury");
  symlinkSync(join(CLI_DIR, target), link);
  return spawnSync(process.execPath, ["--experimental-strip-types", "--no-warnings", link, ...args], {
    encoding: "utf8",
    cwd: dir,
  });
}

test("package bin points at the dedicated bin entry", () => {
  const pkg = JSON.parse(readFileSync(join(PACKAGE_DIR, "package.json"), "utf8")) as { bin: Record<string, string> };
  assert.equal(pkg.bin["llm-jury"], "dist/cli/bin.js");
});

for (const target of ["bin.ts", "main.ts"]) {
  test(`cli runs through a symlink to ${target}: --help`, () => {
    const result = runThroughSymlink(target, ["--help"]);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /Usage: llm-jury <command> \[options\]/);
  });
}

test("cli runs through a symlink: --version prints the library version", () => {
  const result = runThroughSymlink("bin.ts", ["--version"]);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, `${LIBRARY_VERSION}\n`);
});

test("cli usage errors exit with code 2 and a message on stderr", () => {
  const result = runThroughSymlink("bin.ts", [
    "classify",
    "--input",
    "in.jsonl",
    "--output",
    "out.jsonl",
    "--concurrency",
    "ten",
  ]);
  assert.equal(result.status, 2);
  assert.equal(result.stdout, "");
  assert.match(result.stderr, /Error: Invalid value for '--concurrency': must be an integer >= 1, got 'ten'\./);
});
