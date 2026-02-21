import type { DebateTranscript } from "../debate/engine.ts";
import type { JudgeStrategy, Verdict } from "./base.ts";

export class WeightedVoteJudge implements JudgeStrategy {
  async judge(transcript: DebateTranscript, _labels: string[]): Promise<Verdict> {
    const finalRound = transcript.rounds[transcript.rounds.length - 1] ?? [];

    if (finalRound.length === 0) {
      return {
        label: transcript.primaryResult.label,
        confidence: transcript.primaryResult.confidence,
        reasoning: "No persona responses available. Falling back to primary result.",
        wasEscalated: true,
        primaryResult: transcript.primaryResult,
        debateTranscript: transcript,
        judgeStrategy: "weighted_vote",
        totalDurationMs: transcript.durationMs,
        totalCostUsd: transcript.totalCostUsd,
      };
    }

    const scores = new Map<string, number>();
    for (const response of finalRound) {
      scores.set(response.label, (scores.get(response.label) ?? 0) + Number(response.confidence));
    }

    let winner = "";
    let bestScore = -1;
    for (const [label, score] of scores.entries()) {
      if (score > bestScore) {
        winner = label;
        bestScore = score;
      }
    }

    const total = Array.from(scores.values()).reduce((a, b) => a + b, 0) || 1;
    return {
      label: winner,
      confidence: bestScore / total,
      reasoning: "Weighted vote based on persona confidence scores.",
      wasEscalated: true,
      primaryResult: transcript.primaryResult,
      debateTranscript: transcript,
      judgeStrategy: "weighted_vote",
      totalDurationMs: transcript.durationMs,
      totalCostUsd: transcript.totalCostUsd,
    };
  }
}
