import type { ClassificationResult } from "../classifiers/base.ts";
import { DEFAULT_MODEL } from "../defaults.ts";
import { LiteLLMClient } from "../llm/client.ts";
import type { LLMClient } from "../llm/client.ts";
import { NOOP_LOGGER } from "../logger.ts";
import type { Logger } from "../logger.ts";
import type { Persona, PersonaResponse } from "../personas/base.ts";
import { buildPersonaResponseSchema } from "../personas/schema.ts";
import {
  addCosts,
  formatConfidence,
  matchLabel,
  parseConfidence,
  safeJsonObject,
  stripMarkdown,
  wrapUntrusted,
} from "../utils.ts";

/** Absorbs float noise so spend that lands exactly on the cap does not trip it. */
export const COST_CAP_TOLERANCE_USD = 1e-9;

/**
 * Spend the cost guards compare against the cap: the known cost so far plus
 * the per-call estimate for every call that reported no cost, so a client
 * that never reports cost cannot slip past the cap.
 */
export function guardSpend(
  knownCostUsd: number | null | undefined,
  unpricedCalls: number,
  estimatedCostPerCallUsd: number | null | undefined,
): number {
  return (knownCostUsd ?? 0) + unpricedCalls * (estimatedCostPerCallUsd ?? 0);
}

/** True when `spendUsd` is over the cap (null/undefined means no cap). */
export function exceedsCostCap(spendUsd: number, maxCostUsd: number | null | undefined): boolean {
  if (maxCostUsd == null) {
    return false;
  }
  return spendUsd > maxCostUsd + COST_CAP_TOLERANCE_USD;
}

const SUMMARISATION_PROMPT =
  "You are a neutral summarisation agent. You have observed a structured debate " +
  "among domain experts about classifying a piece of text.\n\n" +
  "Produce a concise synthesis that covers:\n" +
  "1. The main arguments from each side\n" +
  "2. Points of consensus among the experts\n" +
  "3. Unresolved disagreements\n\n" +
  "Be factual and impartial. Do not add your own classification.";

const DELIBERATION_INSTRUCTIONS =
  "You have seen the initial assessments from all experts on this input. " +
  "You MUST:\n" +
  "(i) Engage with at least one other expert's reasoning — agree or disagree " +
  "with supporting rationale.\n" +
  "(ii) Revise your own classification if you find their counter-arguments compelling.\n" +
  "(iii) Re-evaluate the input through the interpretive lens of at least one other expert's " +
  "perspective, considering both intent and impact.\n\n" +
  "Then provide your revised assessment.";

export const DebateMode = {
  INDEPENDENT: "independent",
  SEQUENTIAL: "sequential",
  DELIBERATION: "deliberation",
  ADVERSARIAL: "adversarial",
} as const;

export type DebateMode = (typeof DebateMode)[keyof typeof DebateMode];

export type DebateTranscript = {
  inputText: string;
  primaryResult: ClassificationResult;
  rounds: PersonaResponse[][];
  summary?: string;
  durationMs: number;
  totalTokens: number;
  // Sum of the costs reported by the persona and summariser calls; null when
  // no call reported a cost.
  totalCostUsd: number | null;
  // LLM calls in the debate (persona and summariser, including calls that
  // threw) that reported no cost. totalCostUsd leaves them out, so a
  // non-zero count means the total is a lower bound. Always set by
  // DebateEngine; optional so hand-built transcripts still type-check.
  unpricedCalls?: number;
  // Persona name -> knownBias, for the personas that declare one. Always
  // set by DebateEngine.
  personaBiases?: Record<string, string>;
};

/** Responses that carry a real vote (persona call and parse succeeded). */
export function validResponses(responses: PersonaResponse[]): PersonaResponse[] {
  return responses.filter((response) => !response.failed);
}

/** Number of persona calls across all rounds that failed. */
export function countPersonaFailures(rounds: PersonaResponse[][]): number {
  return rounds.reduce(
    (total, round) => total + round.filter((response) => response.failed).length,
    0,
  );
}

export class DebateConfig {
  mode: DebateMode;
  maxRounds: number;
  includePrimaryResult: boolean;
  includeConfidence: boolean;
  // F7: optional high-confidence early stop for the DELIBERATION
  // debate loop. When set, the loop also exits when the MIN
  // persona confidence in a round is >= this threshold, even if
  // personas disagree on label. undefined disables (default
  // behaviour: only unanimous-label consensus halts early).
  earlyStopMinConfidence?: number;

  constructor(options: Partial<DebateConfig> = {}) {
    this.mode = options.mode ?? DebateMode.DELIBERATION;
    this.maxRounds = options.maxRounds ?? 2;
    this.includePrimaryResult = options.includePrimaryResult ?? true;
    this.includeConfidence = options.includeConfidence ?? true;
    this.earlyStopMinConfidence = options.earlyStopMinConfidence;
  }
}

export class DebateEngine {
  private personas: Persona[];
  private config: DebateConfig;
  private llmClient: LLMClient;
  private concurrency: number;
  private logger: Logger;

  constructor(
    personas: Persona[],
    config = new DebateConfig(),
    llmClient: LLMClient = new LiteLLMClient(),
    concurrency = 5,
    logger: Logger = NOOP_LOGGER,
  ) {
    this.personas = personas;
    this.config = config;
    this.llmClient = llmClient;
    this.concurrency = Math.max(1, concurrency);
    this.logger = logger;
  }

  static jsonResponseBlock(): string {
    return (
      "\n## Your Assessment\n\n" +
      "Provide your classification. Respond ONLY with valid JSON:\n" +
      "```json\n" +
      "{\n" +
      '  "label": "<your classification>",\n' +
      '  "confidence": <0.0-1.0>,\n' +
      '  "reasoning": "<your full reasoning>",\n' +
      '  "key_factors": ["<factor 1>", "<factor 2>"],\n' +
      '  "dissent_notes": "<optional rebuttal against opposing side>"\n' +
      "}\n" +
      "```"
    );
  }

  /**
   * Run the debate.
   *
   * `maxCostUsd` caps the debate's spend. Spend is the cost reported so far
   * plus `estimatedCostPerCallUsd` for every call that reported no cost, so
   * the cap still works with clients that never report cost. Once spend is
   * over the cap no new persona batch, round or summariser call starts.
   */
  async debate(
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    maxCostUsd: number | null = null,
    estimatedCostPerCallUsd = 0,
  ): Promise<DebateTranscript> {
    const start = Date.now();
    const rounds: PersonaResponse[][] = [];
    const perCallEstimate = Math.max(0, Number(estimatedCostPerCallUsd) || 0);
    let totalTokens = 0;
    let totalCostUsd: number | null = null;
    let unpricedCalls = 0;

    const recordCall = (tokens: number | undefined, costUsd: number | null | undefined): void => {
      totalTokens += Number(tokens ?? 0);
      if (costUsd == null) {
        unpricedCalls += 1;
      } else {
        totalCostUsd = addCosts(totalCostUsd, costUsd);
      }
    };
    const recordRound = (responses: PersonaResponse[]): void => {
      responses.forEach((response) => recordCall(response.tokensUsed, response.costUsd));
    };
    const spendSoFar = (): number => guardSpend(totalCostUsd, unpricedCalls, perCallEstimate);
    const overBudget = (): boolean => exceedsCostCap(spendSoFar(), maxCostUsd);
    const transcript = (summary?: string): DebateTranscript => ({
      inputText: text,
      primaryResult,
      rounds,
      summary,
      durationMs: Date.now() - start,
      totalTokens,
      totalCostUsd,
      unpricedCalls,
      personaBiases: this.personaBiases(),
    });

    if (this.personas.length === 0) {
      return { ...transcript(), totalCostUsd: 0 };
    }

    if (this.config.mode === DebateMode.INDEPENDENT || this.config.mode === DebateMode.ADVERSARIAL) {
      const responses = await this.runRound(text, primaryResult, labels, [], maxCostUsd, spendSoFar(), perCallEstimate);
      rounds.push(responses);
      recordRound(responses);
    } else if (this.config.mode === DebateMode.SEQUENTIAL) {
      const responses: PersonaResponse[] = [];
      for (const persona of this.personas) {
        let response: PersonaResponse;
        try {
          response = await this.queryPersona(
            persona,
            text,
            primaryResult,
            labels,
            responses.length > 0 ? [responses] : [],
          );
        } catch (err) {
          response = this.failedPersonaResponse(persona, err, labels);
        }
        responses.push(response);
        recordCall(response.tokensUsed, response.costUsd);
        if (overBudget()) {
          break;
        }
      }
      rounds.push(responses);
    } else if (this.config.mode === DebateMode.DELIBERATION) {
      const firstRound = await this.runRound(text, primaryResult, labels, [], maxCostUsd, spendSoFar(), perCallEstimate);
      rounds.push(firstRound);
      recordRound(firstRound);

      if (overBudget()) {
        return transcript();
      }

      // If every persona call failed (bad API key, provider outage),
      // further rounds and the summariser are doomed too — stop paying
      // for them. The failed round stays in the transcript for audit.
      if (validResponses(firstRound).length === 0) {
        this.logger.warn(
          "[llm-jury] all persona calls failed in the opening round; aborting debate early",
          { personas: firstRound.length },
        );
        return transcript();
      }

      // Consensus in the opening round (unanimous labels, or the
      // earlyStopMinConfidence rule) makes further rounds and the summary
      // redundant: the judge gets the opening round only.
      if (this.consensusReached(firstRound)) {
        return transcript();
      }

      for (let i = 1; i < Math.max(1, this.config.maxRounds); i += 1) {
        const current = await this.runDeliberationRound(
          text,
          primaryResult,
          labels,
          rounds,
          maxCostUsd,
          spendSoFar(),
          perCallEstimate,
        );
        rounds.push(current);
        recordRound(current);

        if (overBudget()) {
          break;
        }
        if (validResponses(current).length === 0) {
          this.logger.warn(
            "[llm-jury] all persona calls failed in a deliberation round; halting further rounds",
            { personas: current.length },
          );
          break;
        }
        if (this.consensusReached(current)) {
          break;
        }
      }

      // Stage 3: Summarisation — degrade gracefully if the summariser call
      // fails. The persona rounds are the load-bearing output; a missing
      // synthesis must not crash the verdict.
      let summary: string | undefined;
      if (!overBudget()) {
        try {
          const summaryResult = await this.summarise(text, labels, rounds);
          recordCall(summaryResult.tokens, summaryResult.cost);
          summary = summaryResult.summary;
        } catch (err) {
          // A call that threw reported no cost; count it as unpriced.
          recordCall(0, null);
          this.logger.warn("[llm-jury] summarisation failed; returning transcript without summary", {
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      return transcript(summary);
    }

    return transcript();
  }

  async runRound(
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
    maxCostUsd: number | null = null,
    costSoFar = 0,
    estimatedCostPerCallUsd = 0,
  ): Promise<PersonaResponse[]> {
    return this.runBatched(
      (persona) => this.queryPersona(persona, text, primaryResult, labels, priorRounds),
      labels,
      maxCostUsd,
      costSoFar,
      estimatedCostPerCallUsd,
    );
  }

  async runDeliberationRound(
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
    maxCostUsd: number | null = null,
    costSoFar = 0,
    estimatedCostPerCallUsd = 0,
  ): Promise<PersonaResponse[]> {
    return this.runBatched(
      (persona) => this.queryPersonaDeliberation(persona, text, primaryResult, labels, priorRounds),
      labels,
      maxCostUsd,
      costSoFar,
      estimatedCostPerCallUsd,
    );
  }

  /**
   * Query every persona in batches of `concurrency`. Before each batch the
   * running spend (`costSoFar` plus each response's reported cost, or
   * `estimatedCostPerCallUsd` when it reported none or the call threw) is
   * checked against `maxCostUsd`; once it is over the cap the remaining
   * personas are skipped.
   */
  private async runBatched(
    query: (persona: Persona) => Promise<PersonaResponse>,
    labels: string[],
    maxCostUsd: number | null,
    costSoFar: number,
    estimatedCostPerCallUsd: number,
  ): Promise<PersonaResponse[]> {
    const out: PersonaResponse[] = [];
    let cumulative = costSoFar;
    for (let i = 0; i < this.personas.length; i += this.concurrency) {
      if (exceedsCostCap(cumulative, maxCostUsd)) {
        this.logger.warn("[llm-jury] cost cap reached mid-round; halting remaining personas", {
          cumulativeCostUsd: cumulative,
          maxCostUsd,
          personasRun: out.length,
          personasRemaining: this.personas.length - i,
        });
        break;
      }
      const batch = this.personas.slice(i, i + this.concurrency);
      const settled = await Promise.allSettled(batch.map((persona) => query(persona)));
      settled.forEach((result, idx) => {
        const persona = batch[idx]!;
        if (result.status === "fulfilled") {
          out.push(result.value);
          cumulative += result.value.costUsd ?? estimatedCostPerCallUsd;
        } else {
          out.push(this.failedPersonaResponse(persona, result.reason, labels));
          cumulative += estimatedCostPerCallUsd;
        }
      });
    }
    return out;
  }

  /**
   * Placeholder for a persona whose call threw. `costUsd` is left undefined:
   * the call reported no cost, so the transcript counts it as unpriced.
   */
  failedPersonaResponse(persona: Persona, error: unknown, labels: string[]): PersonaResponse {
    const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    this.logger.warn(`[llm-jury] persona ${persona.name} failed during debate`, { error: message });
    return {
      personaName: persona.name,
      label: labels[0] ?? "unknown",
      confidence: 0,
      reasoning: `Persona call failed: ${message}`,
      keyFactors: [],
      tokensUsed: 0,
      failed: true,
    };
  }

  async queryPersona(
    persona: Persona,
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): Promise<PersonaResponse> {
    const prompt = this.buildPersonaPrompt(persona, text, primaryResult, labels, priorRounds);
    return this.callPersona(persona, prompt, labels);
  }

  async queryPersonaDeliberation(
    persona: Persona,
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): Promise<PersonaResponse> {
    const prompt = this.buildDeliberationPrompt(persona, text, primaryResult, labels, priorRounds);
    return this.callPersona(persona, prompt, labels);
  }

  private async callPersona(persona: Persona, prompt: string, labels: string[]): Promise<PersonaResponse> {
    const schema = buildPersonaResponseSchema(labels) as unknown as Record<string, unknown>;
    const payload = await this.llmClient.complete(persona.model, persona.systemPrompt, prompt, persona.temperature, schema);
    const parsed = this.parsePersonaResponse(payload.content, persona.name, labels);
    if (parsed.failed) {
      this.logger.warn(`[llm-jury] persona ${persona.name} returned an unusable response`, {
        reasoning: parsed.reasoning,
      });
    }
    parsed.rawResponse = payload.content;
    parsed.tokensUsed = Number(payload.tokens ?? 0);
    // Unreported cost stays undefined so the transcript can count the call
    // as unpriced instead of recording a false $0.
    parsed.costUsd = payload.costUsd == null ? undefined : Number(payload.costUsd);
    return parsed;
  }

  buildPersonaPrompt(
    persona: Persona,
    text: string,
    primary: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): string {
    const parts: string[] = [];
    parts.push(`## Persona\n\n${persona.name}: ${persona.role}\n`);
    parts.push(`## Input to Classify\n\n${wrapUntrusted(text)}\n`);
    parts.push(`## Available Labels\n\n${labels.join(", ")}\n`);

    if (this.config.mode === DebateMode.ADVERSARIAL) {
      const personaIndex = this.personas.indexOf(persona);
      const stance = personaIndex % 2 === 0 ? "Prosecution" : "Defense";
      parts.push(
        "## Adversarial Role\n\n" +
          `You are assigned the **${stance}** side. Argue this stance rigorously while staying truthful to the evidence.\n`,
      );
    }

    if (this.config.includePrimaryResult) {
      const confidence = this.config.includeConfidence ? ` (confidence: ${formatConfidence(primary.confidence)})` : "";
      parts.push(
        "## Primary Classifier Result\n\n" +
          `Label: ${primary.label}${confidence}\n` +
          "Note: This was flagged as low-confidence and escalated to you.\n",
      );
    }

    if (priorRounds.length > 0) {
      parts.push("## Previous Assessments\n");
      priorRounds.forEach((roundResponses, idx) => {
        const valid = validResponses(roundResponses);
        if (valid.length === 0) {
          return;
        }
        parts.push(`\n### Round ${idx + 1}\n`);
        for (const response of valid) {
          parts.push(
            `**${response.personaName}**: ${response.label} (confidence: ${response.confidence.toFixed(2)})\n` +
              `Reasoning: ${response.reasoning}\n`,
          );
        }
      });
    }

    parts.push(DebateEngine.jsonResponseBlock());

    return parts.join("\n");
  }

  buildDeliberationPrompt(
    persona: Persona,
    text: string,
    primary: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): string {
    const parts: string[] = [];
    parts.push(`## Persona\n\n${persona.name}: ${persona.role}\n`);
    parts.push(`## Input to Classify\n\n${wrapUntrusted(text)}\n`);
    parts.push(`## Available Labels\n\n${labels.join(", ")}\n`);

    if (this.config.includePrimaryResult) {
      const confidence = this.config.includeConfidence ? ` (confidence: ${formatConfidence(primary.confidence)})` : "";
      parts.push(
        "## Primary Classifier Result\n\n" +
          `Label: ${primary.label}${confidence}\n` +
          "Note: This was flagged as low-confidence and escalated to you.\n",
      );
    }

    if (priorRounds.length > 0) {
      priorRounds.forEach((roundResponses, idx) => {
        const valid = validResponses(roundResponses);
        if (valid.length === 0) {
          return;
        }
        if (idx === 0) {
          parts.push("## Initial Expert Opinions\n");
        } else {
          parts.push(`## Revised Opinions (Round ${idx + 1})\n`);
        }
        for (const response of valid) {
          parts.push(
            `**${response.personaName}**: ${response.label} (confidence: ${response.confidence.toFixed(2)})\n` +
              `Reasoning: ${response.reasoning}\n`,
          );
        }
      });
    }

    parts.push(`\n## Deliberation Instructions\n\n${DELIBERATION_INSTRUCTIONS}\n`);

    parts.push(DebateEngine.jsonResponseBlock());

    return parts.join("\n");
  }

  /**
   * Summarise the debate with one LLM call. `cost` is null when the client
   * reported no cost for the call.
   */
  async summarise(
    text: string,
    labels: string[],
    rounds: PersonaResponse[][],
  ): Promise<{ summary: string; tokens: number; cost: number | null }> {
    const parts: string[] = [];
    parts.push(`## Input\n\n${wrapUntrusted(text)}\n`);
    parts.push(`## Available Labels\n\n${labels.join(", ")}\n`);
    parts.push("## Expert Debate\n");
    rounds.forEach((roundResponses, idx) => {
      const valid = validResponses(roundResponses);
      if (valid.length === 0) {
        return;
      }
      if (idx === 0) {
        parts.push("\n### Initial Expert Opinions\n");
      } else {
        parts.push(`\n### Revised Opinions (Round ${idx + 1})\n`);
      }
      for (const response of valid) {
        parts.push(
          `**${response.personaName}**: ${response.label} (confidence: ${response.confidence.toFixed(2)})\n` +
            `Reasoning: ${response.reasoning}\n`,
        );
      }
    });

    const model = this.personas[0]?.model ?? DEFAULT_MODEL;
    const payload = await this.llmClient.complete(model, SUMMARISATION_PROMPT, parts.join("\n"), 0);

    return {
      summary: payload.content,
      tokens: Number(payload.tokens ?? 0),
      cost: payload.costUsd == null ? null : Number(payload.costUsd),
    };
  }

  /**
   * Parse a persona's raw output. The response is marked `failed` (it stays
   * in the transcript but carries no vote) when the output is not a JSON
   * object, the label does not match one of `labels`, or the confidence is
   * not a finite number. Matched labels are returned in their configured
   * spelling. An empty `labels` accepts any non-empty label.
   */
  parsePersonaResponse(raw: string, personaName: string, labels: string[] = []): PersonaResponse {
    const failed = (reasoning: string): PersonaResponse => ({
      personaName,
      label: "unknown",
      confidence: 0,
      reasoning,
      keyFactors: [],
      failed: true,
    });

    const parsed = safeJsonObject(stripMarkdown(raw));
    if (!parsed) {
      return failed(`Failed to parse persona response: ${raw.slice(0, 200)}`);
    }

    const label = matchLabel(parsed.label, labels);
    if (label === null) {
      return failed(
        `Persona returned label '${String(parsed.label)}', which is not one of the configured labels.`,
      );
    }

    const confidence = parseConfidence(parsed.confidence);
    if (confidence === null) {
      return failed(`Persona returned confidence '${String(parsed.confidence)}', which is not a finite number.`);
    }

    return {
      personaName,
      label,
      confidence,
      reasoning: String(parsed.reasoning ?? ""),
      keyFactors: Array.isArray(parsed.key_factors) ? parsed.key_factors.map(String) : [],
      dissentNotes: parsed.dissent_notes == null ? undefined : String(parsed.dissent_notes),
    };
  }

  consensusReached(roundResponses: PersonaResponse[]): boolean {
    // Failed responses are placeholders, not votes: a round where two
    // personas agree and a third errored is real consensus, and a round
    // of pure failures is not unanimous agreement.
    const valid = validResponses(roundResponses);
    if (valid.length === 0) {
      return false;
    }
    const labels = new Set(valid.map((r) => r.label));
    if (labels.size === 1) {
      return true;
    }
    // F7: high-confidence early stop. When every persona this round
    // is highly confident in its own answer, further deliberation
    // rarely changes the verdict — let the judge break the tie now.
    const threshold = this.config.earlyStopMinConfidence;
    if (threshold !== undefined) {
      const minConfidence = Math.min(...valid.map((r) => r.confidence));
      if (minConfidence >= threshold) {
        return true;
      }
    }
    return false;
  }

  private personaBiases(): Record<string, string> {
    const biases: Record<string, string> = {};
    for (const persona of this.personas) {
      if (persona.knownBias) {
        biases[persona.name] = persona.knownBias;
      }
    }
    return biases;
  }
}
