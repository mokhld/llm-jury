import type { LLMClient } from "../llm/client.ts";
import { LiteLLMClient } from "../llm/client.ts";
import type { DebateTranscript } from "../debate/engine.ts";
import type { JudgeStrategy, Verdict } from "./base.ts";
import { stripMarkdown, safeJsonObject } from "../utils.ts";

export type LLMJudgeOptions = {
  model?: string;
  systemPrompt?: string;
  temperature?: number;
  llmClient?: LLMClient;
};

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

  constructor(options: LLMJudgeOptions = {}) {
    this.model = options.model ?? "gpt-5-mini";
    this.systemPrompt = options.systemPrompt ?? LLMJudge.DEFAULT_SYSTEM_PROMPT;
    this.temperature = options.temperature ?? 0;
    this.llmClient = options.llmClient ?? new LiteLLMClient();
  }

  async judge(transcript: DebateTranscript, labels: string[]): Promise<Verdict> {
    const prompt = this.buildPrompt(transcript, labels);
    const payload = await this.llmClient.complete(this.model, this.systemPrompt, prompt, this.temperature);
    const parsed = safeJsonObject(stripMarkdown(payload.content));
    if (!parsed) {
      return {
        label: String(transcript.primaryResult.label),
        confidence: Number(transcript.primaryResult.confidence),
        reasoning: "LLM judge response was not valid JSON. Falling back to primary result.",
        wasEscalated: true,
        primaryResult: transcript.primaryResult,
        debateTranscript: transcript,
        judgeStrategy: "llm_judge_fallback_invalid_json",
        totalDurationMs: transcript.durationMs,
        totalCostUsd: Number(transcript.totalCostUsd ?? 0) + Number(payload.costUsd ?? 0),
      };
    }

    return {
      label: String(parsed.label ?? transcript.primaryResult.label),
      confidence: Number(parsed.confidence ?? transcript.primaryResult.confidence),
      reasoning: String(parsed.reasoning ?? "LLM judge response."),
      wasEscalated: true,
      primaryResult: transcript.primaryResult,
      debateTranscript: transcript,
      judgeStrategy: "llm_judge",
      totalDurationMs: transcript.durationMs,
      totalCostUsd: Number(transcript.totalCostUsd ?? 0) + Number(payload.costUsd ?? 0),
    };
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
