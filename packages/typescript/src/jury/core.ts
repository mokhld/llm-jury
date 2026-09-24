import type { ClassificationResult, Classifier } from "../classifiers/base.ts";
import {
  DebateConfig,
  DebateEngine,
  DebateMode,
  countPersonaFailures,
  exceedsCostCap,
  guardSpend,
} from "../debate/engine.ts";
import { LiteLLMClient } from "../llm/client.ts";
import type { LLMClient } from "../llm/client.ts";
import type { Persona } from "../personas/base.ts";
import { LLMJudge } from "../judges/llmJudge.ts";
import { Verdict } from "../judges/base.ts";
import type { JudgeStrategy } from "../judges/base.ts";
import { NOOP_LOGGER } from "../logger.ts";
import type { Logger } from "../logger.ts";
import { createSemaphore } from "../utils.ts";

/**
 * Total cost of an escalated verdict: the primary classifier's cost plus the
 * debate (and judge) cost. Null when either part is unknown, so a free
 * primary (cost 0) plus an unpriced debate is reported as unknown, not $0.
 */
function escalatedTotalCost(
  primaryCostUsd: number | null | undefined,
  debateCostUsd: number | null | undefined,
): number | null {
  if (primaryCostUsd == null || debateCostUsd == null) {
    return null;
  }
  return primaryCostUsd + debateCostUsd;
}

export type JuryOptions = {
  classifier: Classifier;
  personas: Persona[];
  confidenceThreshold?: number;
  debateConcurrency?: number;
  judge?: JudgeStrategy;
  debateConfig?: DebateConfig;
  escalationOverride?: (result: ClassificationResult) => boolean;
  maxDebateCostUsd?: number;
  // Estimated cost of one LLM call (persona, summariser or judge). Used for
  // the pre-flight estimate and, for calls whose client reports no cost, by
  // the maxDebateCostUsd guards.
  estimatedCostPerPersonaUsd?: number;
  onEscalation?: (text: string, result: ClassificationResult) => void;
  onCostEstimate?: (estimateUsd: number, text: string) => boolean | undefined;
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
  estimatedCostPerPersonaUsd: number;
  onEscalation?: (text: string, result: ClassificationResult) => void;
  onCostEstimate?: (estimateUsd: number, text: string) => boolean | undefined;
  onVerdict?: (verdict: Verdict) => void;
  logger: Logger;
  private _stats: JuryStats;

  constructor(options: JuryOptions) {
    const threshold = options.confidenceThreshold ?? 0.7;
    if (typeof threshold !== "number" || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      throw new RangeError(`confidenceThreshold must be a finite number in [0, 1]; got ${String(threshold)}.`);
    }
    this.classifier = options.classifier;
    this.personas = options.personas;
    this.threshold = threshold;
    this.logger = options.logger ?? NOOP_LOGGER;
    const llmClient = options.llmClient ?? new LiteLLMClient({ logger: this.logger });
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
    this.estimatedCostPerPersonaUsd = Math.max(0, options.estimatedCostPerPersonaUsd ?? 0.01);
    this.onEscalation = options.onEscalation;
    this.onCostEstimate = options.onCostEstimate;
    this.onVerdict = options.onVerdict;
    this._stats = new JuryStats();
  }

  /**
   * Upper-bound estimate of one escalation's LLM spend: every call the debate
   * and judge can make, times `estimatedCostPerPersonaUsd`. DELIBERATION makes
   * one call per persona per round plus a summariser call; other modes make
   * one call per persona. An LLMJudge adds one call.
   */
  get estimatedMaxDebateCostUsd(): number {
    const deliberation = this.debateConfig.mode === DebateMode.DELIBERATION;
    const personaCalls = this.personas.length * (deliberation ? Math.max(1, this.debateConfig.maxRounds) : 1);
    const summariserCalls = deliberation ? 1 : 0;
    const judgeCalls = this.judge instanceof LLMJudge ? 1 : 0;
    return this.estimatedCostPerPersonaUsd * (personaCalls + summariserCalls + judgeCalls);
  }

  async classify(text: string): Promise<Verdict> {
    const start = Date.now();
    const primary = await this.classifier.classify(text);
    this._stats.total += 1;

    const shouldEscalate = this.shouldEscalate(primary) && this.personas.length > 0;
    if (!shouldEscalate) {
      this._stats.fastPath += 1;
      return this.deliver(
        new Verdict({
          label: primary.label,
          confidence: primary.confidence,
          reasoning: "Classified by primary classifier with sufficient confidence.",
          wasEscalated: false,
          primaryResult: primary,
          debateTranscript: null,
          judgeStrategy: "primary_classifier",
          totalDurationMs: Date.now() - start,
          totalCostUsd: primary.costUsd ?? null,
        }),
      );
    }

    this._stats.escalated += 1;
    this.logger.info("[llm-jury] escalating to debate", {
      label: primary.label,
      confidence: primary.confidence,
    });
    this.onEscalation?.(text, primary);

    // F4: optional user-supplied pre-debate cost gate. Fires before
    // the hardcoded maxDebateCostUsd guard so user logic can
    // short-circuit on policy beyond a fixed cap (per-tenant
    // budgets, time-of-day, etc.). Returning false skips the
    // debate. Returning true or undefined proceeds.
    if (this.onCostEstimate) {
      const decision = this.onCostEstimate(this.estimatedMaxDebateCostUsd, text);
      if (decision === false) {
        this.logger.info("[llm-jury] skipping debate: onCostEstimate returned false", {
          estimateUsd: this.estimatedMaxDebateCostUsd,
        });
        return this.deliver(
          new Verdict({
            label: primary.label,
            confidence: primary.confidence,
            reasoning:
              "Debate skipped: onCostEstimate callback returned false. " +
              "Returning primary classifier result.",
            wasEscalated: true,
            primaryResult: primary,
            debateTranscript: null,
            judgeStrategy: "cost_guard_user_override",
            totalDurationMs: Date.now() - start,
            totalCostUsd: primary.costUsd ?? null,
          }),
        );
      }
    }

    if (this.maxDebateCostUsd != null && exceedsCostCap(this.estimatedMaxDebateCostUsd, this.maxDebateCostUsd)) {
      this.logger.warn("[llm-jury] skipping debate: estimated cost exceeds budget", {
        estimatedCostUsd: this.estimatedMaxDebateCostUsd,
        maxDebateCostUsd: this.maxDebateCostUsd,
      });
      return this.deliver(
        new Verdict({
          label: primary.label,
          confidence: primary.confidence,
          reasoning:
            `Debate skipped: estimated cost (${this.estimatedMaxDebateCostUsd.toFixed(4)} USD) exceeds ` +
            `maxDebateCostUsd (${this.maxDebateCostUsd.toFixed(4)} USD). Returning primary classifier result.`,
          wasEscalated: true,
          primaryResult: primary,
          debateTranscript: null,
          judgeStrategy: "cost_guard_pre_flight",
          totalDurationMs: Date.now() - start,
          totalCostUsd: primary.costUsd ?? null,
        }),
      );
    }

    const transcript = await this.debateEngine.debate(
      text,
      primary,
      this.classifier.labels,
      this.maxDebateCostUsd ?? null,
      this.estimatedCostPerPersonaUsd,
    );

    // Same spend formula as the engine's mid-flight guard: reported cost
    // plus the per-call estimate for every call that reported none.
    const spendUsd = guardSpend(
      transcript.totalCostUsd,
      transcript.unpricedCalls ?? 0,
      this.estimatedCostPerPersonaUsd,
    );
    if (exceedsCostCap(spendUsd, this.maxDebateCostUsd)) {
      this.logger.warn("[llm-jury] debate cost exceeded budget; falling back to primary result", {
        actualCostUsd: transcript.totalCostUsd,
        unpricedCalls: transcript.unpricedCalls ?? 0,
        spendUsd,
        maxDebateCostUsd: this.maxDebateCostUsd,
      });
      return this.deliver(
        new Verdict({
          label: primary.label,
          confidence: primary.confidence,
          reasoning: "Debate exceeded maxDebateCostUsd. Returning primary classifier result.",
          wasEscalated: true,
          primaryResult: primary,
          debateTranscript: transcript,
          judgeStrategy: "cost_guard_primary_fallback",
          totalDurationMs: Date.now() - start,
          totalCostUsd: escalatedTotalCost(primary.costUsd, transcript.totalCostUsd),
          personaFailures: countPersonaFailures(transcript.rounds),
        }),
      );
    }

    const verdict = await this.judge.judge(transcript, this.classifier.labels);
    // Jury is authoritative for wasEscalated and personaFailures: it KNOWS
    // this code path is the escalation branch and it holds the transcript,
    // so judges can't override either.
    verdict.wasEscalated = true;
    verdict.primaryResult = primary;
    verdict.debateTranscript = transcript;
    verdict.totalDurationMs = Date.now() - start;
    // The judge reports debate + judge cost; the verdict total also
    // includes the primary classifier's call.
    verdict.totalCostUsd = escalatedTotalCost(primary.costUsd, verdict.totalCostUsd ?? transcript.totalCostUsd);
    verdict.personaFailures = countPersonaFailures(transcript.rounds);
    if (verdict.personaFailures > 0) {
      this.logger.warn("[llm-jury] verdict is degraded: persona call(s) failed during the debate", {
        personaFailures: verdict.personaFailures,
      });
    }

    return this.deliver(verdict);
  }

  /**
   * Classify many texts concurrently.
   *
   * With `returnExceptions` unset (default) the first failing text rejects
   * the batch, mirroring `Promise.all`, and no further text starts a
   * `classify` call (calls already in flight run to completion). Pass
   * `returnExceptions: true` to receive the `Error` in that text's slot
   * instead, so one bad row cannot discard the verdicts (and spend) of the
   * rows that succeeded.
   */
  async classifyBatch(texts: string[], concurrency?: number, returnExceptions?: false): Promise<Verdict[]>;
  async classifyBatch(texts: string[], concurrency: number | undefined, returnExceptions: true): Promise<Array<Verdict | Error>>;
  async classifyBatch(
    texts: string[],
    concurrency = 10,
    returnExceptions = false,
  ): Promise<Verdict[] | Array<Verdict | Error>> {
    const semaphore = createSemaphore(Math.max(1, concurrency));
    let aborted = false;
    return Promise.all(
      texts.map(async (text) => {
        await semaphore.acquire();
        try {
          if (aborted) {
            // The batch has already rejected; this rejection is never observed.
            throw new Error("classifyBatch aborted after an earlier text failed");
          }
          return await this.classify(text);
        } catch (err) {
          if (!returnExceptions) {
            aborted = true;
            throw err;
          }
          return err instanceof Error ? err : new Error(String(err));
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
    // A missing or non-numeric confidence (NaN, a string, undefined) cannot
    // be trusted to clear the threshold, so it escalates.
    if (typeof result.confidence !== "number" || !Number.isFinite(result.confidence)) {
      return true;
    }
    return result.confidence < this.threshold;
  }

  get stats(): JuryStats {
    return this._stats;
  }

  /** Single exit for every verdict classify() returns, so onVerdict fires exactly once per verdict. */
  private deliver(verdict: Verdict): Verdict {
    this.onVerdict?.(verdict);
    return verdict;
  }
}
