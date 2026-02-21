import type { ClassificationResult } from "../classifiers/base.ts";
import type { DebateTranscript } from "../debate/engine.ts";

export type Verdict = {
  label: string;
  confidence: number;
  reasoning: string;
  wasEscalated: boolean;
  primaryResult: ClassificationResult;
  debateTranscript: DebateTranscript | null;
  judgeStrategy: string;
  totalDurationMs: number;
  totalCostUsd: number | null;
};

export interface JudgeStrategy {
  judge(transcript: DebateTranscript, labels: string[]): Promise<Verdict>;
}
