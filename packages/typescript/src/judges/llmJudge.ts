import type { LLMClient } from "../llm/client.ts";
import { LiteLLMClient } from "../llm/client.ts";
import type { DebateTranscript } from "../debate/engine.ts";
import { DEFAULT_MODEL } from "../defaults.ts";
import { Verdict } from "./base.ts";
import type { JudgeStrategy } from "./base.ts";
import { NOOP_LOGGER } from "../logger.ts";
import type { Logger } from "../logger.ts";
import { stripMarkdown, safeJsonObject } from "../utils.ts";

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
  if ((a == null) && (b == null)) return null;
  return Number(a ?? 0) + Number(b ?? 0);
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
    this.llmClient = options.llmClient ?? new LiteLLMClient();
    this.logger = options.logger ?? NOOP_LOGGER;
  }

  async judge(transcript: DebateTranscript, labels: string[]): Promise<Verdict> {
    const prompt = this.buildPrompt(transcript, labels);
    const payload = await this.llmClient.complete(this.model, this.systemPrompt, prompt, this.temperature);
    const totalCostUsd = sumCosts(transcript.totalCostUsd, payload.costUsd);
    const parsed = safeJsonObject(stripMarkdown(payload.content));
    if (!parsed) {
      this.logger.warn("[llm-jury] LLMJudge response was not valid JSON; falling back to primary result", {
        rawContent: payload.content.slice(0, 500),
      });
      return new Verdict({
        label: String(transcript.primaryResult.label),
        confidence: Number(transcript.primaryResult.confidence),
        reasoning: "LLM judge response was not valid JSON. Falling back to primary result.",
        wasEscalated: true,
        primaryResult: transcript.primaryResult,
        debateTranscript: transcript,
        judgeStrategy: "llm_judge_fallback_invalid_json",
        totalDurationMs: transcript.durationMs,
        totalCostUsd,
      });
    }

    return new Verdict({
      label: String(parsed.label ?? transcript.primaryResult.label),
      confidence: Number(parsed.confidence ?? transcript.primaryResult.confidence),
      reasoning: String(parsed.reasoning ?? "LLM judge response."),
      wasEscalated: true,
      primaryResult: transcript.primaryResult,
      debateTranscript: transcript,
      judgeStrategy: "llm_judge",
      totalDurationMs: transcript.durationMs,
      totalCostUsd,
    });
  }

  buildPrompt(transcript: DebateTranscript, labels: string[]): string {
    const lines: string[] = [];
    lines.push(`Input: ${transcript.inputText}`);
    lines.push(`Available labels: ${labels.join(", ")}`);
    lines.push(
      `Primary result: ${transcript.primaryResult.label} (${Number(transcript.primaryResult.confidence).toFixed(2)})`,
    );
    lines.push("Debate transcript:");

    transcript.rounds.forEach((round, index) => {
      if (index === 0) {
        lines.push("Initial Expert Opinions:");
      } else {
        lines.push(`Revised Opinions (Round ${index + 1}):`);
      }
      round.forEach((response) => {
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
