import type { ClassificationResult, Classifier } from "./base.ts";
import { DEFAULT_MODEL } from "../defaults.ts";
import type { LLMClient } from "../llm/client.ts";
import { LiteLLMClient } from "../llm/client.ts";
import { buildClassifierResponseSchema } from "../personas/schema.ts";
import { matchLabel, parseConfidence, safeJsonObject, stripMarkdown, wrapUntrusted } from "../utils.ts";

export type LLMClassifierOptions = {
  model?: string;
  labels?: string[];
  systemPrompt?: string;
  temperature?: number;
  llmClient?: LLMClient;
};

export class LLMClassifier implements Classifier {
  public labels: string[];
  private model: string;
  private systemPrompt: string;
  private temperature: number;
  private llmClient: LLMClient;

  constructor(options: LLMClassifierOptions = {}) {
    const cleaned = (options.labels ?? [])
      .map((label) => String(label).trim())
      .filter((label) => label.length > 0);
    if (cleaned.length === 0) {
      throw new Error(
        "LLMClassifier requires at least one non-empty label. " +
          "Pass labels: ['safe', 'unsafe'] (or similar) when constructing.",
      );
    }
    this.model = options.model ?? DEFAULT_MODEL;
    this.labels = cleaned;
    this.systemPrompt =
      options.systemPrompt ?? "Classify the text and return JSON with fields label and confidence.";
    this.temperature = options.temperature ?? 0;
    this.llmClient = options.llmClient ?? new LiteLLMClient();
  }

  /**
   * Classify `text` with one LLM call. Output the jury cannot trust (not
   * JSON, a label outside `labels`, or a non-numeric confidence) comes back
   * with confidence 0 so the jury escalates it; `rawOutput.error` says why.
   */
  async classify(text: string): Promise<ClassificationResult> {
    const prompt = [
      "Classify the following text.",
      `Labels: ${this.labels.join(", ") || "any"}`,
      `Text:\n${wrapUntrusted(text)}`,
      "Respond ONLY with JSON: {\"label\":\"...\",\"confidence\":0.0-1.0}",
    ].join("\n");

    const responseFormat = buildClassifierResponseSchema(this.labels) as unknown as Record<string, unknown>;
    const payload = await this.llmClient.complete(
      this.model,
      this.systemPrompt,
      prompt,
      this.temperature,
      responseFormat,
    );
    const parsed = safeJsonObject(stripMarkdown(payload.content));
    const fallbackLabel = this.labels[0] ?? "unknown";
    const callCost = payload.costUsd ?? null;
    if (!parsed) {
      return {
        label: fallbackLabel,
        confidence: 0,
        rawOutput: { raw_content: payload.content, error: "invalid_json" },
        costUsd: callCost,
      };
    }

    const label = matchLabel(parsed.label, this.labels);
    if (label === null) {
      return {
        label: fallbackLabel,
        confidence: 0,
        rawOutput: { raw_content: payload.content, error: "label_not_in_labels" },
        costUsd: callCost,
      };
    }

    const confidence = parseConfidence(parsed.confidence);
    if (confidence === null) {
      return {
        label,
        confidence: 0,
        rawOutput: { raw_content: payload.content, error: "invalid_confidence" },
        costUsd: callCost,
      };
    }

    return {
      label,
      confidence,
      rawOutput: parsed,
      costUsd: callCost,
    };
  }

  async classifyBatch(texts: string[]): Promise<ClassificationResult[]> {
    const out: ClassificationResult[] = [];
    for (const text of texts) {
      out.push(await this.classify(text));
    }
    return out;
  }
}
