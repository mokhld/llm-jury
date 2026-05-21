import test from "node:test";
import assert from "node:assert/strict";

import { FunctionClassifier } from "../../src/classifiers/functionAdapter.ts";
import { DebateConfig, DebateEngine, DebateMode } from "../../src/debate/engine.ts";
import { Jury } from "../../src/jury/core.ts";
import { LLMJudge } from "../../src/judges/llmJudge.ts";
import { MajorityVoteJudge } from "../../src/judges/majorityVote.ts";
import { NOOP_LOGGER } from "../../src/logger.ts";
import type { Logger } from "../../src/logger.ts";
import type { Persona } from "../../src/personas/base.ts";
import { PersonaRegistry } from "../../src/personas/registry.ts";
import { FakeLLMClient } from "../helpers.ts";

type LogEntry = { level: "debug" | "info" | "warn" | "error"; message: string; meta?: unknown };

function createSpyLogger(): { logger: Logger; entries: LogEntry[] } {
  const entries: LogEntry[] = [];
  const logger: Logger = {
    debug: (message, meta) => entries.push({ level: "debug", message, meta }),
    info: (message, meta) => entries.push({ level: "info", message, meta }),
    warn: (message, meta) => entries.push({ level: "warn", message, meta }),
    error: (message, meta) => entries.push({ level: "error", message, meta }),
  };
  return { logger, entries };
}

test("NOOP_LOGGER discards calls and is the default", async () => {
  // If the default were not silent, this would produce console output during tests.
  // We assert that the jury constructs and runs without any user-visible side effect.
  const classifier = new FunctionClassifier(() => ["safe", 0.95], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [] });
  assert.equal(jury.logger, NOOP_LOGGER);
  const verdict = await jury.classify("hi");
  assert.equal(verdict.wasEscalated, false);
});

test("logger.info fires on escalation", async () => {
  const { logger, entries } = createSpyLogger();
  const classifier = new FunctionClassifier(() => ["unknown", 0.3], ["safe", "unsafe"]);
  const llm = new FakeLLMClient();
  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.contentModeration(),
    judge: new MajorityVoteJudge(),
    llmClient: llm,
    logger,
  });

  await jury.classify("ambiguous");

  const escalationLog = entries.find((e) => e.level === "info" && e.message.includes("escalating"));
  assert.ok(escalationLog, "expected an info log when escalating");
});

test("logger.warn fires on cost-guard fallback", async () => {
  const { logger, entries } = createSpyLogger();
  const classifier = new FunctionClassifier(() => ["safe", 0.3], ["safe", "unsafe"]);
  const llm = new FakeLLMClient({
    "Policy Analyst": { content: JSON.stringify({ label: "unsafe", confidence: 0.9, reasoning: "x", key_factors: [] }), costUsd: 0.5 },
    "Cultural Context Expert": { content: JSON.stringify({ label: "safe", confidence: 0.5, reasoning: "y", key_factors: [] }), costUsd: 0.5 },
    "Harm Assessment Specialist": { content: JSON.stringify({ label: "unsafe", confidence: 0.8, reasoning: "z", key_factors: [] }), costUsd: 0.5 },
  });
  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.contentModeration(),
    llmClient: llm,
    maxDebateCostUsd: 0.6,
    logger,
  });

  await jury.classify("text");

  const costLog = entries.find(
    (e) => e.level === "warn" && e.message.includes("cost exceeded"),
  );
  assert.ok(costLog, "expected a warn log when cost guard fires");
});

test("logger.warn fires on persona failure during debate", async () => {
  const { logger, entries } = createSpyLogger();
  const personas: Persona[] = [
    { name: "A", role: "r", systemPrompt: "A", model: "gpt-5-mini", temperature: 0.3 },
    { name: "B", role: "r", systemPrompt: "B", model: "gpt-5-mini", temperature: 0.3 },
  ];

  class FlakyClient {
    async complete(_model: string, systemPrompt: string, _prompt: string, _t = 0) {
      if (systemPrompt === "B") {
        throw new Error("simulated failure");
      }
      return {
        content: JSON.stringify({ label: "safe", confidence: 0.8, reasoning: "ok", key_factors: [] }),
        tokens: 10,
        costUsd: 0.001,
      };
    }
  }

  const engine = new DebateEngine(
    personas,
    new DebateConfig({ mode: DebateMode.INDEPENDENT }),
    new FlakyClient(),
    5,
    logger,
  );

  await engine.debate("text", { label: "x", confidence: 0.3 }, ["safe", "unsafe"]);

  const failureLog = entries.find(
    (e) => e.level === "warn" && e.message.includes("persona B failed"),
  );
  assert.ok(failureLog, "expected a warn log when a persona fails");
});

test("logger.warn fires when LLMJudge response is unparseable", async () => {
  const { logger, entries } = createSpyLogger();
  const llm = new FakeLLMClient({ "judge-model": { content: "totally not json" } });
  const judge = new LLMJudge({ model: "judge-model", llmClient: llm, logger });

  await judge.judge(
    {
      inputText: "x",
      primaryResult: { label: "safe", confidence: 0.4 },
      rounds: [[
        { personaName: "A", label: "safe", confidence: 0.5, reasoning: "r", keyFactors: [] },
      ]],
      durationMs: 1,
      totalTokens: 1,
      totalCostUsd: 0,
    },
    ["safe", "unsafe"],
  );

  const parseLog = entries.find(
    (e) => e.level === "warn" && e.message.includes("not valid JSON"),
  );
  assert.ok(parseLog, "expected a warn log when LLMJudge JSON parse fails");
});

test("console satisfies the Logger interface", () => {
  // Compile-time and runtime check: passing console must not throw.
  const classifier = new FunctionClassifier(() => ["safe", 0.95], ["safe", "unsafe"]);
  const jury = new Jury({ classifier, personas: [], logger: console as Logger });
  assert.equal(jury.logger, console);
});
