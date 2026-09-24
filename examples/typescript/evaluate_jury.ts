/**
 * Measure whether the jury beats the primary classifier on labelled data.
 *
 * Runs offline: the personas use a stub LLM client that answers from a
 * lookup table, so no API key is needed. In production, drop `llmClient` and
 * the jury uses its default client (and real money, so set `maxEscalations`).
 *
 * Run with:
 *   node --experimental-strip-types examples/typescript/evaluate_jury.ts
 */
import {
  DebateConfig,
  DebateMode,
  FunctionClassifier,
  Jury,
  JuryEvaluator,
  MajorityVoteJudge,
} from "@llm-jury/core";
import type { LLMClient } from "@llm-jury/core";

// text, true label, primary label, primary confidence, what the stub jury answers
const DATA: Array<[string, string, string, number, string]> = [
  ["Thanks for the quick reply, see you Monday.", "safe", "safe", 0.98, "safe"],
  ["Buy cheap followers now, limited offer!!!", "unsafe", "unsafe", 0.97, "unsafe"],
  ["I will find out where you live.", "unsafe", "safe", 0.62, "unsafe"],
  ["This movie absolutely killed it.", "safe", "unsafe", 0.58, "safe"],
  ["Great, another Monday. Kill me now.", "safe", "unsafe", 0.66, "safe"],
  ["Send me your password to verify your account.", "unsafe", "unsafe", 0.81, "unsafe"],
  ["The recipe calls for a pinch of salt.", "safe", "safe", 0.93, "safe"],
  ["You people are the worst, get out.", "unsafe", "unsafe", 0.72, "unsafe"],
  ["Meet me behind the gym after school.", "safe", "unsafe", 0.55, "unsafe"],
  ["Nice shot, you really destroyed them.", "safe", "safe", 0.77, "unsafe"],
  ["Click here to claim your prize.", "unsafe", "safe", 0.9, "unsafe"],
  ["Happy birthday, have a great one!", "safe", "safe", 0.99, "safe"],
];

/** Answers every persona call from a lookup table, at a made-up cost. */
class StubLLMClient implements LLMClient {
  private answers: Map<string, string>;

  constructor(answers: Map<string, string>) {
    this.answers = answers;
  }

  async complete(_model: string, _systemPrompt: string, prompt: string) {
    const label = [...this.answers].find(([text]) => prompt.includes(text))?.[1] ?? "safe";
    const content = JSON.stringify({ label, confidence: 0.8, reasoning: "stub" });
    return { content, tokens: 50, costUsd: 0.0004 };
  }
}

async function main(): Promise<void> {
  const texts = DATA.map((row) => row[0]);
  const labels = DATA.map((row) => row[1]);
  const predictions = new Map(DATA.map((row): [string, [string, number]] => [row[0], [row[2], row[3]]]));

  const jury = new Jury({
    classifier: new FunctionClassifier((text) => predictions.get(text)!, ["safe", "unsafe"]),
    personas: ["policy analyst", "context reader", "harm assessor"].map((name) => ({
      name,
      role: name,
      systemPrompt: `You are the ${name}.`,
      model: "stub-model",
      temperature: 0,
    })),
    judge: new MajorityVoteJudge(),
    debateConfig: new DebateConfig({ mode: DebateMode.INDEPENDENT }),
    llmClient: new StubLLMClient(new Map(DATA.map((row) => [row[0], row[4]]))),
  });

  const report = await new JuryEvaluator(jury).evaluate({
    texts,
    labels,
    bandUpper: 0.95,
    maxEscalations: 50,
  });
  const summary = report.summary();
  const fixed = (value: number | null, digits = 2) => (value === null ? "n/a" : value.toFixed(digits));
  console.log(`Items: ${summary.n}, debated: ${summary.debated}`);
  console.log(`Primary accuracy:             ${fixed(summary.primaryAccuracy)}`);
  console.log(`Primary accuracy on debated:  ${fixed(summary.primaryAccuracyOnDebated)}`);
  console.log(`Jury accuracy on debated:     ${fixed(summary.juryAccuracyOnDebated)}`);
  console.log(`Flips helped / hurt:          ${summary.flipsHelped} / ${summary.flipsHurt}`);
  console.log(`Mean debate cost:             $${fixed(summary.meanDebateCostUsd, 4)}`);
  console.log();

  for (const row of report.thresholdSweep({ errorCost: 10 })) {
    console.log(
      `  threshold=${row.threshold.toFixed(2)}  ` +
        `escalation_rate=${fixed(row.escalationRate)}  ` +
        `system_accuracy=${fixed(row.systemAccuracy)}  ` +
        `errors=${row.errors}  ` +
        `cost=${row.totalCost.toFixed(2)}`,
    );
  }
  console.log();
  console.log(`Best threshold: ${report.bestThreshold({ errorCost: 10 })}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
