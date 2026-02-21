import type { DebateTranscript } from "../debate/engine.ts";
import type { JudgeStrategy, Verdict } from "./base.ts";

export type PersonaPriors = Record<string, Record<string, number>>;

export class BayesianJudge implements JudgeStrategy {
  private priors: PersonaPriors;

  constructor(priors: PersonaPriors = {}) {
    this.priors = priors;
  }

  async judge(transcript: DebateTranscript, labels: string[]): Promise<Verdict> {
    const finalRound = transcript.rounds[transcript.rounds.length - 1] ?? [];
    if (finalRound.length === 0) {
      return {
        label: transcript.primaryResult.label,
        confidence: transcript.primaryResult.confidence,
        reasoning: "No persona responses available. Falling back to primary result.",
        wasEscalated: true,
        primaryResult: transcript.primaryResult,
        debateTranscript: transcript,
        judgeStrategy: "bayesian",
        totalDurationMs: transcript.durationMs,
        totalCostUsd: transcript.totalCostUsd,
      };
    }

    const posterior = new Map<string, number>();
    labels.forEach((label) => posterior.set(label, 1));

    for (const response of finalRound) {
      for (const label of labels) {
        const prior = this.priors[response.personaName]?.[label] ?? 1 / Math.max(1, labels.length);
        const likelihood = response.label === label ? Number(response.confidence) : 1 - Number(response.confidence);
        posterior.set(label, (posterior.get(label) ?? 1) * Math.max(1e-6, prior * likelihood));
      }
    }

    const sum = Array.from(posterior.values()).reduce((a, b) => a + b, 0) || 1;
    let winner = labels[0] ?? transcript.primaryResult.label;
    let winnerScore = -1;

    for (const [label, score] of posterior.entries()) {
      const normalized = score / sum;
      if (normalized > winnerScore) {
        winner = label;
        winnerScore = normalized;
      }
    }

    return {
      label: winner,
      confidence: winnerScore,
      reasoning: "Bayesian aggregation across persona responses.",
      wasEscalated: true,
      primaryResult: transcript.primaryResult,
      debateTranscript: transcript,
      judgeStrategy: "bayesian",
      totalDurationMs: transcript.durationMs,
      totalCostUsd: transcript.totalCostUsd,
    };
  }
}
