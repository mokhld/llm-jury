import type { LLMClient } from "../llm/client.ts";
import { LiteLLMClient } from "../llm/client.ts";
import { validResponses } from "../debate/engine.ts";
import type { DebateTranscript } from "../debate/engine.ts";
import { DEFAULT_MODEL } from "../defaults.ts";
import { buildJudgeResponseSchema } from "../personas/schema.ts";
import { Verdict, fallbackVerdict } from "./base.ts";
import type { JudgeStrategy } from "./base.ts";
import { MajorityVoteJudge } from "./majorityVote.ts";
import { NOOP_LOGGER } from "../logger.ts";
import type { Logger } from "../logger.ts";
import {
  addCosts,
  formatConfidence,
  matchLabel,
  parseConfidence,
  safeJsonObject,
  stripMarkdown,
  wrapUntrusted,
} from "../utils.ts";

export type LLMJudgeOptions = {
  model?: string;
  systemPrompt?: string;
  temperature?: number;
  llmClient?: LLMClient;
  logger?: Logger;
};

/**
 * Sum two cost components, preserving null when both are unknown.
 * Returns null only when both inputs are null/undefined; otherwise treats
 * the unknown component as 0. This avoids silently reporting `0` when cost
 * tracking actually failed.
 */
export function sumCosts(a: number | null | undefined, b: number | null | undefined): number | null {
  return addCosts(a, b);
}

function toStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

export class LLMJudge implements JudgeStrategy {
  static readonly DEFAULT_SYSTEM_PROMPT =
    "You are the presiding judge in an expert panel. " +
    "You have received assessments from multiple domain experts on a classification task.\n\n" +
    "Your role is to:\n" +
    "1. Weigh each expert's reasoning on its merits\n" +
    "2. Consider the strength of evidence each expert cites\n" +
    "3. Note where experts agree and disagree\n" +
    "4. Factor in each expert's known perspective/bias\n" +
    "5. If a debate summary is provided, use it to identify the decisive arguments\n" +
    "6. Deliver a final classification with clear reasoning\n\n" +
    "Respond ONLY with valid JSON:\n" +
    "{\n" +
    '  "label": "<final classification>",\n' +
    '  "confidence": <0.0-1.0>,\n' +
    '  "reasoning": "<your synthesis of the debate>",\n' +
    '  "key_agreements": ["<points all experts agreed on>"],\n' +
    '  "key_disagreements": ["<points of contention>"],\n' +
    '  "decisive_factor": "<what tipped the decision>"\n' +
    "}";

  private model: string;
  private systemPrompt: string;
  private temperature: number;
  private llmClient: LLMClient;
  private logger: Logger;

  constructor(options: LLMJudgeOptions = {}) {
    this.model = options.model ?? DEFAULT_MODEL;
    this.systemPrompt = options.systemPrompt ?? LLMJudge.DEFAULT_SYSTEM_PROMPT;
    this.temperature = options.temperature ?? 0;
    this.logger = options.logger ?? NOOP_LOGGER;
    this.llmClient = options.llmClient ?? new LiteLLMClient({ logger: this.logger });
  }

  async judge(transcript: DebateTranscript, labels: string[]): Promise<Verdict> {
    // If every persona call failed there is nothing to judge — skip the
    // LLM call (it would only see failure placeholders) and fall back to
    // the primary classifier result.
    if (!transcript.rounds.some((round) => validResponses(round).length > 0)) {
      return fallbackVerdict(
        transcript,
        "llm_judge_fallback_personas_failed",
        "All persona calls failed; returning primary classifier result.",
      );
    }

    const prompt = this.buildPrompt(transcript, labels);
    const responseFormat = buildJudgeResponseSchema(labels) as unknown as Record<string, unknown>;

    // The debate is already paid for, so a judge outage (5xx after retries,
    // timeout, auth error) degrades to a vote instead of rejecting classify().
    let payload: Awaited<ReturnType<LLMClient["complete"]>>;
    try {
      payload = await this.llmClient.complete(
        this.model,
        this.systemPrompt,
        prompt,
        this.temperature,
        responseFormat,
      );
    } catch (err) {
      const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      this.logger.warn("[llm-jury] LLMJudge call failed; falling back to a majority vote of the final round", {
        error: message,
      });
      return this.voteFallback(
        transcript,
        labels,
        "llm_judge_fallback_error",
        `LLM judge call failed (${message}).`,
        transcript.totalCostUsd,
      );
    }

    const totalCostUsd = addCosts(transcript.totalCostUsd, payload.costUsd);
    const parsed = safeJsonObject(stripMarkdown(payload.content));
    if (!parsed) {
      this.logger.warn(
        "[llm-jury] LLMJudge response was not valid JSON; falling back to a majority vote of the final round",
        { rawContent: payload.content.slice(0, 500) },
      );
      return this.voteFallback(
        transcript,
        labels,
        "llm_judge_fallback_invalid_json",
        "LLM judge response was not valid JSON.",
        totalCostUsd,
      );
    }

    const label = matchLabel(parsed.label, labels);
    if (label === null) {
      this.logger.warn(
        "[llm-jury] LLMJudge returned a label outside the configured labels; falling back to a majority vote of the final round",
        { label: parsed.label, labels },
      );
      return this.voteFallback(
        transcript,
        labels,
        "llm_judge_fallback_invalid_label",
        `LLM judge returned label '${String(parsed.label)}', which is not one of the configured labels.`,
        totalCostUsd,
      );
    }

    const confidence = parseConfidence(parsed.confidence);
    if (confidence === null) {
      this.logger.warn(
        "[llm-jury] LLMJudge returned an invalid confidence; falling back to a majority vote of the final round",
        { confidence: parsed.confidence },
      );
      return this.voteFallback(
        transcript,
        labels,
        "llm_judge_fallback_invalid_confidence",
        `LLM judge returned confidence '${String(parsed.confidence)}', which is not a finite number.`,
        totalCostUsd,
      );
    }

    return new Verdict({
      label,
      confidence,
      reasoning: String(parsed.reasoning ?? "LLM judge response."),
      wasEscalated: true,
      primaryResult: transcript.primaryResult,
      debateTranscript: transcript,
      judgeStrategy: "llm_judge",
      totalDurationMs: transcript.durationMs,
      totalCostUsd,
      judgeDetails: {
        keyAgreements: toStringList(parsed.key_agreements),
        keyDisagreements: toStringList(parsed.key_disagreements),
        decisiveFactor: parsed.decisive_factor == null ? null : String(parsed.decisive_factor),
      },
    });
  }

  /**
   * Verdict used when the judge's own output is unusable: a majority vote
   * over the final round's valid responses (no further LLM call), or the
   * primary classifier result when that round has no valid responses.
   */
  private async voteFallback(
    transcript: DebateTranscript,
    labels: string[],
    judgeStrategy: string,
    reason: string,
    totalCostUsd: number | null,
  ): Promise<Verdict> {
    const lastRound = transcript.rounds[transcript.rounds.length - 1] ?? [];
    if (validResponses(lastRound).length === 0) {
      const verdict = fallbackVerdict(
        transcript,
        judgeStrategy,
        `${reason} No valid persona responses in the final round; returning primary classifier result.`,
      );
      verdict.totalCostUsd = totalCostUsd;
      return verdict;
    }

    const vote = await new MajorityVoteJudge().judge(transcript, labels);
    vote.reasoning =
      `${reason} Falling back to a majority vote over the final round's persona responses. ${vote.reasoning}`;
    vote.judgeStrategy = judgeStrategy;
    vote.totalCostUsd = totalCostUsd;
    return vote;
  }

  buildPrompt(transcript: DebateTranscript, labels: string[]): string {
    const lines: string[] = [];
    lines.push(`Input:\n${wrapUntrusted(transcript.inputText)}`);
    lines.push(`Available labels: ${labels.join(", ")}`);
    lines.push(
      `Primary result: ${transcript.primaryResult.label} (${formatConfidence(transcript.primaryResult.confidence)})`,
    );

    const biases = Object.entries(transcript.personaBiases ?? {});
    if (biases.length > 0) {
      lines.push("Expert roster:");
      for (const [name, bias] of biases) {
        lines.push(`- ${name} (known bias: ${bias})`);
      }
    }

    lines.push("Debate transcript:");

    transcript.rounds.forEach((round, index) => {
      const valid = validResponses(round);
      if (valid.length === 0) {
        return;
      }
      if (index === 0) {
        lines.push("Initial Expert Opinions:");
      } else {
        lines.push(`Revised Opinions (Round ${index + 1}):`);
      }
      valid.forEach((response) => {
        lines.push(
          `- ${response.personaName}: ${response.label} (${Number(response.confidence).toFixed(2)}) | Reasoning: ${response.reasoning}`,
        );
      });
    });

    if (transcript.summary) {
      lines.push("");
      lines.push("Debate Summary:");
      lines.push(transcript.summary);
    }

    lines.push(
      "Respond ONLY with JSON containing: label, confidence, reasoning, key_agreements, key_disagreements, decisive_factor.",
    );

    return lines.join("\n");
  }
}
