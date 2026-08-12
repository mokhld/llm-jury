import test from "node:test";
import assert from "node:assert/strict";

import { DebateConfig, DebateEngine, DebateMode, countPersonaFailures } from "../../src/debate/engine.ts";
import type { Persona, PersonaResponse } from "../../src/personas/base.ts";

class AlwaysFailingClient {
  public calls = 0;

  async complete(): Promise<{ content: string; tokens: number; costUsd: number }> {
    this.calls += 1;
    throw new Error("no api key");
  }
}

function makePersonas(): Persona[] {
  return ["A", "B", "C"].map((name) => ({
    name,
    role: "role",
    systemPrompt: name,
    model: "gpt-fake",
    temperature: 0,
  }));
}

const primary = { label: "unknown", confidence: 0.4 };

test("failed persona calls produce responses marked failed", async () => {
  const llm = new AlwaysFailingClient();
  const engine = new DebateEngine(
    makePersonas(),
    new DebateConfig({ mode: DebateMode.INDEPENDENT }),
    llm,
  );

  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);

  for (const response of transcript.rounds[0]!) {
    assert.equal(response.failed, true);
  }
  assert.equal(countPersonaFailures(transcript.rounds), 3);
});

test("unparseable persona output is marked failed", () => {
  const engine = new DebateEngine(makePersonas());
  const response = engine.parsePersonaResponse("not json", "A");
  assert.equal(response.failed, true);
});

test("deliberation aborts after an all-failed opening round", async () => {
  const llm = new AlwaysFailingClient();
  const engine = new DebateEngine(
    makePersonas(),
    new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 3 }),
    llm,
  );

  const transcript = await engine.debate("text", primary, ["safe", "unsafe"]);

  // Only the opening round ran: 3 persona calls, no deliberation rounds,
  // no summariser call.
  assert.equal(transcript.rounds.length, 1);
  assert.equal(llm.calls, 3);
});

test("consensus ignores failed responses", () => {
  const engine = new DebateEngine(makePersonas());
  const failed: PersonaResponse = {
    personaName: "C",
    label: "safe",
    confidence: 0,
    reasoning: "failed",
    keyFactors: [],
    failed: true,
  };
  const realA: PersonaResponse = { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "r", keyFactors: [] };
  const realB: PersonaResponse = { personaName: "B", label: "unsafe", confidence: 0.8, reasoning: "r", keyFactors: [] };

  // Two real agreeing votes + one failed placeholder = consensus.
  assert.equal(engine.consensusReached([realA, realB, failed]), true);
  // A round of pure failures is not unanimous agreement on labels[0].
  assert.equal(engine.consensusReached([failed, failed]), false);
});

test("deliberation prompt excludes failed responses", () => {
  const personas = makePersonas();
  const engine = new DebateEngine(personas, new DebateConfig({ mode: DebateMode.DELIBERATION }));
  const prior: PersonaResponse[][] = [[
    { personaName: "A", label: "unsafe", confidence: 0.9, reasoning: "real reasoning", keyFactors: [] },
    {
      personaName: "B",
      label: "safe",
      confidence: 0,
      reasoning: "Persona call failed: boom",
      keyFactors: [],
      failed: true,
    },
  ]];

  const prompt = engine.buildDeliberationPrompt(personas[0]!, "text", primary, ["safe", "unsafe"], prior);
  assert.match(prompt, /real reasoning/);
  assert.doesNotMatch(prompt, /Persona call failed/);
});
