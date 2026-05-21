import type { ClassificationResult, Classifier } from "../classifiers/base.ts";
import { DebateConfig, DebateEngine } from "../debate/engine.ts";
import { LiteLLMClient } from "../llm/client.ts";
import type { LLMClient } from "../llm/client.ts";
import type { Persona } from "../personas/base.ts";
import { LLMJudge } from "../judges/llmJudge.ts";
import { Verdict } from "../judges/base.ts";
import type { JudgeStrategy } from "../judges/base.ts";
import { NOOP_LOGGER } from "../logger.ts";
import type { Logger } from "../logger.ts";
import { createSemaphore } from "../utils.ts";

export type JuryOptions = {
  classifier: Classifier;
  personas: Persona[];
  confidenceThreshold?: number;
  debateConcurrency?: number;
  judge?: JudgeStrategy;
  debateConfig?: DebateConfig;
  escalationOverride?: (result: ClassificationResult) => boolean;
  maxDebateCostUsd?: number;
  onEscalation?: (text: string, result: ClassificationResult) => void;
  onVerdict?: (verdict: Verdict) => void;
  llmClient?: LLMClient;
  logger?: Logger;
};

export class JuryStats {
  total = 0;
  fastPath = 0;
  escalated = 0;

  get escalationRate(): number {
    return this.total > 0 ? this.escalated / this.total : 0;
  }

  get costSavingsVsAlwaysEscalate(): number {
    return this.total > 0 ? this.fastPath / this.total : 0;
  }
}

export class Jury {
  classifier: Classifier;
  personas: Persona[];
  threshold: number;
  judge: JudgeStrategy;
  debateConfig: DebateConfig;
  debateEngine: DebateEngine;
  escalationOverride?: (result: ClassificationResult) => boolean;
  maxDebateCostUsd?: number;
  onEscalation?: (text: string, result: ClassificationResult) => void;
  onVerdict?: (verdict: Verdict) => void;
  logger: Logger;
  private _stats: JuryStats;

  constructor(options: JuryOptions) {
    this.classifier = options.classifier;
    this.personas = options.personas;
    this.threshold = options.confidenceThreshold ?? 0.7;
    this.logger = options.logger ?? NOOP_LOGGER;
    const llmClient = options.llmClient ?? new LiteLLMClient();
    this.judge = options.judge ?? new LLMJudge({ llmClient, logger: this.logger });
    this.debateConfig = options.debateConfig ?? new DebateConfig();
    this.debateEngine = new DebateEngine(
      this.personas,
      this.debateConfig,
      llmClient,
      Math.max(1, options.debateConcurrency ?? 5),
      this.logger,
    );
    this.escalationOverride = options.escalationOverride;
    this.maxDebateCostUsd = options.maxDebateCostUsd;
    this.onEscalation = options.onEscalation;
    this.onVerdict = options.onVerdict;
    this._stats = new JuryStats();
  }

  async classify(text: string): Promise<Verdict> {
    const start = Date.now();
    const primary = await this.classifier.classify(text);
    this._stats.total += 1;

    const shouldEscalate = this.shouldEscalate(primary) && this.personas.length > 0;
    if (!shouldEscalate) {
      this._stats.fastPath += 1;
      return new Verdict({
        label: primary.label,
        confidence: primary.confidence,
        reasoning: "Classified by primary classifier with sufficient confidence.",
        wasEscalated: false,
        primaryResult: primary,
        debateTranscript: null,
        judgeStrategy: "primary_classifier",
        totalDurationMs: Date.now() - start,
        totalCostUsd: 0,
      });
    }

    this._stats.escalated += 1;
    this.logger.info("[llm-jury] escalating to debate", {
      label: primary.label,
      confidence: primary.confidence,
    });
    this.onEscalation?.(text, primary);

    const transcript = await this.debateEngine.debate(
      text,
      primary,
      this.classifier.labels,
      this.maxDebateCostUsd ?? null,
    );

    if (this.maxDebateCostUsd != null && transcript.totalCostUsd != null && transcript.totalCostUsd > this.maxDebateCostUsd) {
      this.logger.warn("[llm-jury] debate cost exceeded budget; falling back to primary result", {
        actualCostUsd: transcript.totalCostUsd,
        maxDebateCostUsd: this.maxDebateCostUsd,
      });
      return new Verdict({
        label: primary.label,
        confidence: primary.confidence,
        reasoning: "Debate exceeded maxDebateCostUsd. Returning primary classifier result.",
        wasEscalated: true,
        primaryResult: primary,
        debateTranscript: transcript,
        judgeStrategy: "cost_guard_primary_fallback",
        totalDurationMs: Date.now() - start,
        totalCostUsd: transcript.totalCostUsd,
      });
    }

    const verdict = await this.judge.judge(transcript, this.classifier.labels);
    verdict.wasEscalated = true;
    verdict.primaryResult = primary;
    verdict.debateTranscript = transcript;
    verdict.totalDurationMs = Date.now() - start;
    if (verdict.totalCostUsd == null) {
      verdict.totalCostUsd = transcript.totalCostUsd;
    }

    this.onVerdict?.(verdict);
    return verdict;
  }

  async classifyBatch(texts: string[], concurrency = 10): Promise<Verdict[]> {
    const semaphore = createSemaphore(Math.max(1, concurrency));
    return Promise.all(
      texts.map(async (text) => {
        await semaphore.acquire();
        try {
          return await this.classify(text);
        } finally {
          semaphore.release();
        }
      }),
    );
  }

  shouldEscalate(result: ClassificationResult): boolean {
    if (this.escalationOverride) {
      return Boolean(this.escalationOverride(result));
    }
    return result.confidence < this.threshold;
  }

  get stats(): JuryStats {
    return this._stats;
  }
}
