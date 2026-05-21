/**
 * Custom persona definitions with deliberation mode.
 *
 * Requires OPENAI_API_KEY in your environment.
 *
 * Run with:
 *   node --experimental-strip-types examples/typescript/custom_personas.ts
 */
import {
  DebateConfig,
  DebateMode,
  Jury,
  LLMClassifier,
  MajorityVoteJudge,
  type Persona,
} from "@llm-jury/core";

async function main(): Promise<void> {
  const personas: Persona[] = [
    {
      name: "Strict Classifier",
      role: "Applies policy literally",
      systemPrompt: "Prioritize hard policy matches over contextual uncertainty.",
      model: "gpt-5-mini",
      temperature: 0.3,
    },
    {
      name: "Context Advocate",
      role: "Looks for benign interpretations",
      systemPrompt: "Focus on intent, quotation, and contextual nuance.",
      model: "gpt-5-mini",
      temperature: 0.3,
    },
    {
      name: "Harm Sentinel",
      role: "Prioritizes minimizing harmful outcomes",
      systemPrompt: "Focus on downstream harm even under uncertainty.",
      model: "gpt-5-mini",
      temperature: 0.3,
    },
  ];

  const classifier = new LLMClassifier({ labels: ["allow", "review", "reject"] });

  const jury = new Jury({
    classifier,
    personas,
    confidenceThreshold: 0.8,
    judge: new MajorityVoteJudge(),
    debateConfig: new DebateConfig({ mode: DebateMode.DELIBERATION, maxRounds: 2 }),
  });

  const verdict = await jury.classify("That user is trash and should be removed.");

  console.log(`Label:      ${verdict.label}`);
  console.log(`Confidence: ${verdict.confidence.toFixed(2)}`);
  console.log(`Escalated:  ${verdict.wasEscalated}`);
  console.log(`Strategy:   ${verdict.judgeStrategy}`);

  if (verdict.debateTranscript) {
    const rounds = verdict.debateTranscript.rounds;
    console.log(`\nRounds: ${rounds.length}`);
    rounds.forEach((rnd, i) => {
      console.log(`  Round ${i + 1}:`);
      for (const resp of rnd) {
        console.log(`    ${resp.personaName}: ${resp.label} (${resp.confidence.toFixed(2)})`);
      }
    });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
