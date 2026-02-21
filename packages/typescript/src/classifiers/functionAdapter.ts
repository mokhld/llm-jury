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
    return {
      label,
      confidence,
      rawOutput: { label, confidence },
    };
  }

  async classifyBatch(texts: string[]): Promise<ClassificationResult[]> {
    return defaultClassifyBatch(this, texts);
  }
}
