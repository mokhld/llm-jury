/**
 * Content moderation with LLM classifier and persona debate.
 *
 * Requires OPENAI_API_KEY in your environment.
 *
 * Run with:
 *   node --experimental-strip-types examples/typescript/content_moderation.ts
 */
import {
  Jury,
  LLMClassifier,
  MajorityVoteJudge,
  PersonaRegistry,
} from "@llm-jury/core";

async function main(): Promise<void> {
  const classifier = new LLMClassifier({ labels: ["safe", "unsafe"] });

  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.contentModeration(),
    confidenceThreshold: 0.85,
    judge: new MajorityVoteJudge(),
  });

  const verdict = await jury.classify("Those people always ruin everything.");

  console.log(`Label:      ${verdict.label}`);
  console.log(`Confidence: ${verdict.confidence.toFixed(2)}`);
  console.log(`Escalated:  ${verdict.wasEscalated}`);
  console.log(`Strategy:   ${verdict.judgeStrategy}`);
  console.log(`Reasoning:  ${verdict.reasoning.slice(0, 200)}`);

  if (verdict.debateTranscript) {
    const rounds = verdict.debateTranscript.rounds;
    console.log(`\nDebate (${rounds.length} round(s)):`);
    for (const resp of rounds[rounds.length - 1] ?? []) {
      console.log(`  ${resp.personaName}: ${resp.label} (${resp.confidence.toFixed(2)})`);
      console.log(`    ${resp.reasoning.slice(0, 100)}`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
