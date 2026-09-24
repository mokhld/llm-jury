import test from "node:test";
import assert from "node:assert/strict";

import type { ClassificationResult, Classifier } from "../../src/classifiers/base.ts";
import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { DebateConfig, DebateMode } from "../../src/debate/engine.ts";
import { Jury } from "../../src/jury/core.ts";
import { LLMJudge } from "../../src/judges/llmJudge.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import type { Persona } from "../../src/personas/base.ts";

const labels = ["safe", "unsafe"];

function makePersonas(count: number): Persona[] {
  return Array.from({ length: count }, (_, i) => ({
    name: `P${i}`,
    role: "role",
    systemPrompt: `P${i}`,
    model: "persona-model",
    temperature: 0,
  }));
}

class StaticClassifier implements Classifier {
  labels = labels;
  private result: ClassificationResult;
  constructor(result: ClassificationResult) {
    this.result = result;
  }
  async classify(): Promise<ClassificationResult> {
    return { ...this.result };
  }
}

/**
 * Personas alternate "unsafe" / "safe" so no consensus is reached. Cost is
 * reported only for system prompts listed in `costs`; the judge (model
 * "judge") reports `judgeCost` when set.
 */
class Client {
  calls = 0;
  private costs: Record<string, number>;
  private judgeCost?: number;
  private judgeError?: Error;

  constructor(costs: Record<string, number> = {}, judgeCost?: number, judgeError?: Error) {
    this.costs = costs;
    this.judgeCost = judgeCost;
    this.judgeError = judgeError;
  }

  async complete(model: string, systemPrompt: string) {
    this.calls += 1;
    if (model === "judge") {
      if (this.judgeError) throw this.judgeError;
      const content = JSON.stringify({
        label: "unsafe",
        confidence: 0.9,
        reasoning: "judge",
        key_agreements: [],
        key_disagreements: [],
        decisive_factor: "x",
      });
      return this.judgeCost === undefined ? { content } : { content, costUsd: this.judgeCost };
    }
    if (systemPrompt.startsWith("You are a neutral summarisation agent")) {
      return { content: "summary" };
    }
    const index = Number(systemPrompt.slice(1));
    const content = JSON.stringify({
      label: index % 2 === 0 ? "unsafe" : "safe",
      confidence: 0.8,
      reasoning: "r",
      key_factors: [],
    });
    const cost = this.costs[systemPrompt];
    return cost === undefined ? { content } : { content, costUsd: cost };
  }
}

test("mid-flight guard trips when reported plus estimated spend exceeds the cap", async () => {
  // P0 reports 0.05, P1 reports nothing (estimated at 0.01): 0.06 > 0.05
  // after the first batch, so P2 and P3 never run.
  const llm = new Client({ P0: 0.05 });
  const jury = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.3, costUsd: 0.002 }),
    personas: makePersonas(4),
    judge: new MajorityVoteJudge(),
    llmClient: llm,
    debateConcurrency: 2,
    maxDebateCostUsd: 0.05,
    estimatedCostPerPersonaUsd: 0.01,
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
  });

  const verdict = await jury.classify("text");

  assert.equal(llm.calls, 2);
  assert.equal(verdict.judgeStrategy, "cost_guard_primary_fallback");
  assert.equal(verdict.debateTranscript!.unpricedCalls, 1);
  assert.equal(verdict.debateTranscript!.totalCostUsd, 0.05);
  // Primary classifier cost + reported debate cost.
  assert.ok(Math.abs(verdict.totalCostUsd! - 0.052) < 1e-12);
});

test("an unpriced debate reports null cost instead of $0", async () => {
  const llm = new Client();
  const jury = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.3 }),
    personas: makePersonas(3),
    judge: new LLMJudge({ model: "judge", llmClient: llm }),
    llmClient: llm,
    debateConfig: new DebateConfig({ maxRounds: 2 }),
  });

  const verdict = await jury.classify("text");

  assert.equal(verdict.judgeStrategy, "llm_judge");
  assert.equal(verdict.totalCostUsd, null);
  assert.equal(verdict.debateTranscript!.totalCostUsd, null);
  // 3 personas x 2 rounds + summariser.
  assert.equal(verdict.debateTranscript!.unpricedCalls, 7);
  const dict = verdict.toDict();
  assert.equal((dict.debateTranscript as { unpricedCalls: number }).unpricedCalls, 7);
});

test("a free primary plus an unpriced debate is reported as unknown, not $0", async () => {
  const llm = new Client();
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["safe", 0.3], labels),
    personas: makePersonas(3),
    judge: new MajorityVoteJudge(),
    llmClient: llm,
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
  });

  const verdict = await jury.classify("text");

  assert.equal(verdict.primaryResult.costUsd, 0);
  assert.equal(verdict.judgeStrategy, "majority_vote");
  assert.equal(verdict.totalCostUsd, null);
});

test("a skipped debate reports a free primary's cost as 0", async () => {
  const jury = new Jury({
    classifier: new FunctionClassifier(() => ["safe", 0.3], labels),
    personas: makePersonas(3),
    llmClient: new Client(),
    maxDebateCostUsd: 0.001,
  });

  const verdict = await jury.classify("text");

  assert.equal(verdict.judgeStrategy, "cost_guard_pre_flight");
  assert.equal(verdict.totalCostUsd, 0);
});

test("pre-flight estimate tolerates float noise at an exact cap", async () => {
  const llm = new Client();
  const jury = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.3 }),
    personas: makePersonas(3),
    judge: new MajorityVoteJudge(),
    llmClient: llm,
    maxDebateCostUsd: 0.3,
    // 3 x 0.1 = 0.30000000000000004 in floating point.
    estimatedCostPerPersonaUsd: 0.1,
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
  });

  const verdict = await jury.classify("text");

  assert.notEqual(verdict.judgeStrategy, "cost_guard_pre_flight");
  assert.equal(llm.calls, 3);
});

test("judged verdict cost includes the primary classifier, the debate and the judge", async () => {
  const llm = new Client({ P0: 0.01, P1: 0.02 }, 0.004);
  const jury = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.3, costUsd: 0.003 }),
    personas: makePersonas(2),
    judge: new LLMJudge({ model: "judge", llmClient: llm }),
    llmClient: llm,
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
  });

  const verdict = await jury.classify("text");

  assert.equal(verdict.judgeStrategy, "llm_judge");
  assert.ok(Math.abs(verdict.totalCostUsd! - (0.003 + 0.01 + 0.02 + 0.004)) < 1e-12);
});

test("fast path reports the primary cost as given, including unknown", async () => {
  const unknownCost = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.95 }),
    personas: makePersonas(1),
    llmClient: new Client(),
  });
  assert.equal((await unknownCost.classify("t")).totalCostUsd, null);

  const knownCost = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.95, costUsd: 0.002 }),
    personas: makePersonas(1),
    llmClient: new Client(),
  });
  assert.equal((await knownCost.classify("t")).totalCostUsd, 0.002);
});

test("skipped debates report the primary cost", async () => {
  const override = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.3, costUsd: 0.002 }),
    personas: makePersonas(2),
    llmClient: new Client(),
    onCostEstimate: () => false,
  });
  const skipped = await override.classify("t");
  assert.equal(skipped.judgeStrategy, "cost_guard_user_override");
  assert.equal(skipped.totalCostUsd, 0.002);

  const preFlight = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.3 }),
    personas: makePersonas(2),
    llmClient: new Client(),
    maxDebateCostUsd: 0.001,
  });
  const refused = await preFlight.classify("t");
  assert.equal(refused.judgeStrategy, "cost_guard_pre_flight");
  assert.equal(refused.totalCostUsd, null);
});

test("a failing LLM judge does not reject classify()", async () => {
  const llm = new Client({}, undefined, Object.assign(new Error("LLM request failed (503): down"), { status: 503 }));
  const jury = new Jury({
    classifier: new StaticClassifier({ label: "safe", confidence: 0.3 }),
    personas: makePersonas(3),
    judge: new LLMJudge({ model: "judge", llmClient: llm }),
    llmClient: llm,
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
  });

  const verdict = await jury.classify("text");

  assert.equal(verdict.judgeStrategy, "llm_judge_fallback_error");
  // P0 and P2 vote "unsafe", P1 votes "safe".
  assert.equal(verdict.label, "unsafe");
  assert.equal(verdict.wasEscalated, true);
});
