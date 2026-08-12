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
  personaFailures?: number;
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
  // Number of persona calls across the debate that failed (LLM error or
  // unparseable output). Set authoritatively by Jury after judging.
  personaFailures: number;
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
    this.personaFailures = init.personaFailures ?? 0;
    this.libraryVersion = init.libraryVersion ?? LIBRARY_VERSION;
    this.createdAt = init.createdAt ?? new Date().toISOString();
  }

  /**
   * True when at least one persona failed during the debate. Degraded
   * verdicts were decided by fewer jurors than configured (or by none, in
   * which case the primary classifier result was returned). Production
   * pipelines can use this to route to human review.
   */
  get debateDegraded(): boolean {
    return this.personaFailures > 0;
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
      personaFailures: this.personaFailures,
      debateDegraded: this.debateDegraded,
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

export const ALL_FAILED_REASON =
  "All persona calls in the final round failed; returning primary classifier result.";

export function fallbackVerdict(
  transcript: DebateTranscript,
  judgeStrategy: string,
  reason = "No persona responses available. Falling back to primary result.",
): Verdict {
  return new Verdict({
    label: transcript.primaryResult.label,
    confidence: transcript.primaryResult.confidence,
    reasoning: reason,
    wasEscalated: true,
    primaryResult: transcript.primaryResult,
    debateTranscript: transcript,
    judgeStrategy,
    totalDurationMs: transcript.durationMs,
    totalCostUsd: transcript.totalCostUsd,
  });
}
