import type { ClassificationResult, Classifier } from "./base.ts";
import type { LLMClient } from "../llm/client.ts";
import { LiteLLMClient } from "../llm/client.ts";
import { safeJsonObject, stripMarkdown } from "../utils.ts";

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
    this.model = options.model ?? "gpt-5-mini";
    this.labels = options.labels ?? [];
    this.systemPrompt =
      options.systemPrompt ?? "Classify the text and return JSON with fields label and confidence.";
    this.temperature = options.temperature ?? 0;
    this.llmClient = options.llmClient ?? new LiteLLMClient();
  }

  async classify(text: string): Promise<ClassificationResult> {
    const prompt = [
      "Classify the following text.",
      `Labels: ${this.labels.join(", ") || "any"}`,
      `Text: ${text}`,
      "Respond ONLY with JSON: {\"label\":\"...\",\"confidence\":0.0-1.0}",
    ].join("\n");

    const payload = await this.llmClient.complete(this.model, this.systemPrompt, prompt, this.temperature);
    const parsed = safeJsonObject(stripMarkdown(payload.content));
    const fallbackLabel = this.labels[0] ?? "unknown";
    if (!parsed) {
      return {
        label: fallbackLabel,
        confidence: 0,
        rawOutput: { raw_content: payload.content, error: "invalid_json" },
      };
    }

    return {
      label: String(parsed.label ?? fallbackLabel),
      confidence: Number(parsed.confidence ?? 0),
      rawOutput: parsed,
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
