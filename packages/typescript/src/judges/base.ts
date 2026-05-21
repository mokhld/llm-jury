import type { ClassificationResult } from "../classifiers/base.ts";
import type { DebateTranscript } from "../debate/engine.ts";
import { LIBRARY_VERSION } from "../_version.ts";

export type VerdictInit = {
  label: string;
  confidence: number;
  reasoning: string;
  wasEscalated: boolean;
  primaryResult: ClassificationResult;
  debateTranscript: DebateTranscript | null;
  judgeStrategy: string;
  totalDurationMs: number;
  totalCostUsd: number | null;
  libraryVersion?: string;
  createdAt?: string;
};

export class Verdict {
  label: string;
  confidence: number;
  reasoning: string;
  wasEscalated: boolean;
  primaryResult: ClassificationResult;
  debateTranscript: DebateTranscript | null;
  judgeStrategy: string;
  totalDurationMs: number;
  totalCostUsd: number | null;
  libraryVersion: string;
  createdAt: string;

  constructor(init: VerdictInit) {
    this.label = init.label;
    this.confidence = init.confidence;
    this.reasoning = init.reasoning;
    this.wasEscalated = init.wasEscalated;
    this.primaryResult = init.primaryResult;
    this.debateTranscript = init.debateTranscript;
    this.judgeStrategy = init.judgeStrategy;
    this.totalDurationMs = init.totalDurationMs;
    this.totalCostUsd = init.totalCostUsd;
    this.libraryVersion = init.libraryVersion ?? LIBRARY_VERSION;
    this.createdAt = init.createdAt ?? new Date().toISOString();
  }

  toDict(): Record<string, unknown> {
    return {
      label: this.label,
      confidence: this.confidence,
      reasoning: this.reasoning,
      wasEscalated: this.wasEscalated,
      primaryResult: this.primaryResult,
      debateTranscript: this.debateTranscript,
      judgeStrategy: this.judgeStrategy,
      totalDurationMs: this.totalDurationMs,
      totalCostUsd: this.totalCostUsd,
      libraryVersion: this.libraryVersion,
      createdAt: this.createdAt,
    };
  }

  toJSON(): Record<string, unknown> {
    return this.toDict();
  }
}

export interface JudgeStrategy {
  judge(transcript: DebateTranscript, labels: string[]): Promise<Verdict>;
}

export function fallbackVerdict(transcript: DebateTranscript, judgeStrategy: string): Verdict {
  return new Verdict({
    label: transcript.primaryResult.label,
    confidence: transcript.primaryResult.confidence,
    reasoning: "No persona responses available. Falling back to primary result.",
    wasEscalated: true,
    primaryResult: transcript.primaryResult,
    debateTranscript: transcript,
    judgeStrategy,
    totalDurationMs: transcript.durationMs,
    totalCostUsd: transcript.totalCostUsd,
  });
}
