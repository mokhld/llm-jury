import test from "node:test";
import assert from "node:assert/strict";

import type { ClassificationResult } from "../../src/classifiers/base.ts";
import { DebateConfig, DebateEngine, DebateMode } from "../../src/debate/engine.ts";
import type { Persona } from "../../src/personas/base.ts";

const labels = ["safe", "unsafe"];
const primary: ClassificationResult = { label: "safe", confidence: 0.4 };

function makePersonas(count: number): Persona[] {
  return Array.from({ length: count }, (_, i) => ({
    name: `P${i}`,
    role: "role",
    systemPrompt: `P${i}`,
    model: "m",
    temperature: 0,
  }));
}

/**
 * Persona i answers "safe" when i is even, "unsafe" when odd (so
 * deliberation never reaches consensus), with `costs[systemPrompt]` as its
 * reported cost (omitted when undefined). The summariser reports
 * `summaryCost`.
 */
class CostClient {
  calls = 0;
  private costs: Record<string, number | undefined>;
  private summaryCost?: number;

  constructor(costs: Record<string, number | undefined> = {}, summaryCost?: number) {
    this.costs = costs;
    this.summaryCost = summaryCost;
  }

  async complete(_model: string, systemPrompt: string) {
    this.calls += 1;
    if (systemPrompt.startsWith("You are a neutral summarisation agent")) {
      return this.summaryCost === undefined ? { content: "summary" } : { content: "summary", costUsd: this.summaryCost };
    }
    const index = Number(systemPrompt.slice(1));
    const content = JSON.stringify({
      label: index % 2 === 0 ? "safe" : "unsafe",
      confidence: 0.8,
      reasoning: "r",
      key_factors: [],
    });
    const cost = this.costs[systemPrompt];
    return cost === undefined ? { content, tokens: 3 } : { content, tokens: 3, costUsd: cost };
  }
}

test("calls that report no cost leave totalCostUsd null and are counted as unpriced", async () => {
  const llm = new CostClient();
  const engine = new DebateEngine(makePersonas(3), new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 2 }), llm);
  const transcript = await engine.debate("text", primary, labels);
  // 3 personas x 2 rounds + 1 summariser.
  assert.equal(llm.calls, 7);
  assert.equal(transcript.totalCostUsd, null);
  assert.equal(transcript.unpricedCalls, 7);
  assert.equal(transcript.rounds[0]![0]!.costUsd, undefined);
});

test("mixed reporting sums the known costs and counts the rest", async () => {
  const llm = new CostClient({ P0: 0.01, P2: 0.02 }, 0.005);
  const engine = new DebateEngine(makePersonas(3), new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 2 }), llm);
  const transcript = await engine.debate("text", primary, labels);
  assert.ok(Math.abs(transcript.totalCostUsd! - (2 * (0.01 + 0.02) + 0.005)) < 1e-12);
  assert.equal(transcript.unpricedCalls, 2, "P1 in both rounds");
});

test("mid-flight guard counts unpriced calls at the per-call estimate", async () => {
  const llm = new CostClient();
  const engine = new DebateEngine(makePersonas(6), new DebateConfig({ mode: DebateMode.INDEPENDENT }), llm, 2);
  // Batch 1 (2 unpriced calls) is estimated at 0.04 > 0.03, so batches 2
  // and 3 never start.
  const transcript = await engine.debate("text", primary, labels, 0.03, 0.02);
  assert.equal(llm.calls, 2);
  assert.equal(transcript.rounds[0]!.length, 2);
  assert.equal(transcript.unpricedCalls, 2);
});

test("mid-flight guard skips deliberation rounds and the summariser once over budget", async () => {
  const llm = new CostClient({ P0: 0.05 });
  const engine = new DebateEngine(makePersonas(3), new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 3 }), llm);
  // Opening round: 0.05 known + 2 unpriced x 0.01 = 0.07 > 0.06.
  const transcript = await engine.debate("text", primary, labels, 0.06, 0.01);
  assert.equal(llm.calls, 3);
  assert.equal(transcript.rounds.length, 1);
  assert.equal(transcript.summary, undefined);
});

test("sequential mode stops once the estimated spend is over the cap", async () => {
  const llm = new CostClient();
  const engine = new DebateEngine(makePersonas(5), new DebateConfig({ mode: DebateMode.SEQUENTIAL }), llm);
  const transcript = await engine.debate("text", primary, labels, 0.025, 0.01);
  // Spend after each call: 0.01, 0.02, 0.03 > 0.025 -> stop.
  assert.equal(llm.calls, 3);
  assert.equal(transcript.rounds[0]!.length, 3);
});

test("the guard tolerates float noise at an exact cap", async () => {
  const llm = new CostClient({ P0: 0.1, P1: 0.1, P2: 0.1, P3: 0.1 });
  const engine = new DebateEngine(makePersonas(4), new DebateConfig({ mode: DebateMode.INDEPENDENT }), llm, 1);
  // After 3 calls the sum is 0.30000000000000004, which is "at" the cap.
  const transcript = await engine.debate("text", primary, labels, 0.3);
  assert.equal(llm.calls, 4);
  assert.equal(transcript.rounds[0]!.length, 4);
});

test("persona and summariser calls that throw count as unpriced", async () => {
  const client = {
    calls: 0,
    async complete(_model: string, systemPrompt: string) {
      this.calls += 1;
      if (systemPrompt === "P1" || systemPrompt.startsWith("You are a neutral summarisation agent")) {
        throw new Error("upstream down");
      }
      const index = Number(systemPrompt.slice(1));
      return {
        content: JSON.stringify({ label: index % 2 === 0 ? "safe" : "unsafe", confidence: 0.8, reasoning: "r", key_factors: [] }),
        costUsd: 0.01,
      };
    },
  };
  const personas = makePersonas(4);
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 2 }), client);
  const transcript = await engine.debate("text", primary, labels);
  // P1 throws in both rounds, the summariser throws once.
  assert.equal(transcript.unpricedCalls, 3);
  assert.ok(Math.abs(transcript.totalCostUsd! - 6 * 0.01) < 1e-12);
  const failed = transcript.rounds[0]!.find((r) => r.personaName === "P1")!;
  assert.equal(failed.failed, true);
  assert.equal(failed.costUsd, undefined);
});

test("mid-flight guard counts thrown calls at the per-call estimate", async () => {
  let calls = 0;
  const client = {
    async complete(): Promise<{ content: string }> {
      calls += 1;
      throw new Error("no api key");
    },
  };
  const engine = new DebateEngine(makePersonas(6), new DebateConfig({ mode: DebateMode.INDEPENDENT }), client, 2);
  const transcript = await engine.debate("text", primary, labels, 0.03, 0.02);
  assert.equal(calls, 2);
  assert.equal(transcript.unpricedCalls, 2);
  assert.equal(transcript.totalCostUsd, null);
});

test("transcript records each persona's known bias", async () => {
  const personas: Persona[] = [
    { name: "Strict", role: "r", systemPrompt: "P0", model: "m", temperature: 0, knownBias: "policy-strict" },
    { name: "Plain", role: "r", systemPrompt: "P1", model: "m", temperature: 0 },
  ];
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.INDEPENDENT }), new CostClient());
  const transcript = await engine.debate("text", primary, labels);
  assert.deepEqual(transcript.personaBiases, { Strict: "policy-strict" });
});
