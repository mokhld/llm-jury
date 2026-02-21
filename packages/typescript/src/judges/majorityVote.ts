import type { DebateTranscript } from "../debate/engine.ts";
import type { JudgeStrategy, Verdict } from "./base.ts";

export class MajorityVoteJudge implements JudgeStrategy {
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
        judgeStrategy: "majority_vote",
        totalDurationMs: transcript.durationMs,
        totalCostUsd: transcript.totalCostUsd,
      };
    }

    const counts = new Map<string, number>();
    for (const response of finalRound) {
      counts.set(response.label, (counts.get(response.label) ?? 0) + 1);
    }

    let winner = "";
    let winnerCount = -1;
    for (const [label, count] of counts.entries()) {
      if (count > winnerCount) {
        winner = label;
        winnerCount = count;
      }
    }

    const reasoning = finalRound
      .filter((response) => response.label === winner)
      .map((response) => response.reasoning)
      .join(" ");

    return {
      label: winner,
      confidence: winnerCount / finalRound.length,
      reasoning: reasoning || "Majority vote selected the winner.",
      wasEscalated: true,
      primaryResult: transcript.primaryResult,
      debateTranscript: transcript,
      judgeStrategy: "majority_vote",
      totalDurationMs: transcript.durationMs,
      totalCostUsd: transcript.totalCostUsd,
    };
  }
}
