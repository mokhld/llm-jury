import { defaultClassifyBatch } from "./base.ts";
import type { ClassificationResult, Classifier } from "./base.ts";

export class FunctionClassifier implements Classifier {
  public labels: string[];
  private fn: (text: string) => [string, number] | Promise<[string, number]>;

  constructor(fn: (text: string) => [string, number] | Promise<[string, number]>, labels: string[]) {
    this.fn = fn;
    this.labels = labels;
  }

  async classify(text: string): Promise<ClassificationResult> {
    const [label, confidence] = await this.fn(text);
    // A local function makes no paid call, so its cost is a known zero.
    return {
      label,
      confidence,
      rawOutput: { label, confidence },
      costUsd: 0,
    };
  }

  async classifyBatch(texts: string[]): Promise<ClassificationResult[]> {
    return defaultClassifyBatch(this, texts);
  }
}
