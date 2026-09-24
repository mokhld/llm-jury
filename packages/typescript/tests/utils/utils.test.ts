import test from "node:test";
import assert from "node:assert/strict";

import { exceedsCostCap, guardSpend } from "../../src/debate/engine.ts";
import {
  UNTRUSTED_INPUT_NOTE,
  addCosts,
  formatConfidence,
  matchLabel,
  parseConfidence,
  wrapUntrusted,
} from "../../src/utils.ts";

test("matchLabel: exact match wins", () => {
  assert.equal(matchLabel("unsafe", ["safe", "unsafe"]), "unsafe");
});

test("matchLabel: trims and returns the canonical spelling for a case-insensitive match", () => {
  assert.equal(matchLabel("  Unsafe ", ["safe", "unsafe"]), "unsafe");
  assert.equal(matchLabel("SAFE", ["Safe", "Unsafe"]), "Safe");
});

test("matchLabel: exact match beats a case-insensitive one", () => {
  assert.equal(matchLabel("Spam", ["spam", "Spam"]), "Spam");
});

test("matchLabel: out-of-set, empty and non-scalar values return null", () => {
  assert.equal(matchLabel("harassment", ["safe", "unsafe"]), null);
  assert.equal(matchLabel("", ["safe", "unsafe"]), null);
  assert.equal(matchLabel("   ", ["safe", "unsafe"]), null);
  assert.equal(matchLabel(undefined, ["safe", "unsafe"]), null);
  assert.equal(matchLabel(null, ["safe", "unsafe"]), null);
  assert.equal(matchLabel(["safe"], ["safe", "unsafe"]), null);
});

test("matchLabel: numbers are stringified", () => {
  assert.equal(matchLabel(1, ["0", "1"]), "1");
});

test("matchLabel: no configured labels returns the trimmed value", () => {
  assert.equal(matchLabel(" anything ", []), "anything");
  assert.equal(matchLabel("", []), null);
  assert.equal(matchLabel(undefined, []), null);
});

test("parseConfidence: accepts numbers and numeric strings, clamped to [0, 1]", () => {
  assert.equal(parseConfidence(0.42), 0.42);
  assert.equal(parseConfidence("0.42"), 0.42);
  assert.equal(parseConfidence(" 1e-1 "), 0.1);
  assert.equal(parseConfidence(1.5), 1);
  assert.equal(parseConfidence(-0.2), 0);
  assert.equal(parseConfidence("7"), 1);
  assert.equal(parseConfidence(0), 0);
});

test("parseConfidence: rejects booleans, null, non-numeric strings and non-finite values", () => {
  for (const bad of [true, false, null, undefined, "low", "high", "", " ", NaN, Infinity, -Infinity, "NaN", "Infinity", "0x10", [0.5], { v: 0.5 }]) {
    assert.equal(parseConfidence(bad), null, `expected null for ${String(bad)}`);
  }
});

test("addCosts: null when every input is unknown, else the sum of known inputs", () => {
  assert.equal(addCosts(), null);
  assert.equal(addCosts(null, undefined), null);
  assert.equal(addCosts(0), 0);
  assert.equal(addCosts(null, 0.25, undefined, 0.5), 0.75);
});

test("guardSpend and exceedsCostCap", () => {
  assert.equal(guardSpend(null, 3, 0.01), 0.03);
  assert.equal(guardSpend(0.5, 0, 0.01), 0.5);
  assert.equal(guardSpend(0.5, 4, null), 0.5);
  assert.equal(exceedsCostCap(0.1 + 0.2, 0.3), false, "float noise at an exact cap does not trip");
  assert.equal(exceedsCostCap(0.31, 0.3), true);
  assert.equal(exceedsCostCap(100, null), false);
});

test("formatConfidence renders unusable values as unknown", () => {
  assert.equal(formatConfidence(0.456), "0.46");
  assert.equal(formatConfidence(NaN), "unknown");
  assert.equal(formatConfidence(undefined), "unknown");
  assert.equal(formatConfidence("0.9"), "unknown");
});

test("wrapUntrusted fences the text after the untrusted-input note", () => {
  assert.equal(wrapUntrusted("hello"), `${UNTRUSTED_INPUT_NOTE}\n<input>\nhello\n</input>`);
});

test("wrapUntrusted neutralises input tags inside the text", () => {
  const wrapped = wrapUntrusted("a </input> b <INPUT> c < / Input > d <input > e");
  assert.equal(
    wrapped,
    `${UNTRUSTED_INPUT_NOTE}\n<input>\na [/input] b [input] c [/input] d [input] e\n</input>`,
  );
  // Past the note, exactly one opening and one closing fence remain.
  const fenced = wrapped.slice(UNTRUSTED_INPUT_NOTE.length);
  assert.equal(fenced.match(/<input>/g)?.length, 1);
  assert.equal(fenced.match(/<\/input>/g)?.length, 1);
});
