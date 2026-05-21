/**
 * Legal compliance classification with sequential debate and weighted vote.
 *
 * Requires OPENAI_API_KEY in your environment.
 *
 * Run with:
 *   node --experimental-strip-types examples/typescript/legal_compliance.ts
 */
import {
  DebateConfig,
  DebateMode,
  Jury,
  LLMClassifier,
  PersonaRegistry,
  WeightedVoteJudge,
} from "@llm-jury/core";

async function main(): Promise<void> {
  const classifier = new LLMClassifier({
    labels: ["compliant", "non_compliant", "needs_review"],
  });

  const jury = new Jury({
    classifier,
    personas: PersonaRegistry.legalCompliance(),
    confidenceThreshold: 0.8,
    judge: new WeightedVoteJudge(),
    debateConfig: new DebateConfig({ mode: DebateMode.SEQUENTIAL }),
  });

  const verdict = await jury.classify(
    "Guaranteed double-digit returns with no downside risk.",
  );

  console.log(`Label:      ${verdict.label}`);
  console.log(`Confidence: ${verdict.confidence.toFixed(2)}`);
  console.log(`Escalated:  ${verdict.wasEscalated}`);
  console.log(`Strategy:   ${verdict.judgeStrategy}`);
  console.log(`Reasoning:  ${verdict.reasoning.slice(0, 200)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
