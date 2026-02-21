import type { ClassificationResult, Classifier } from "./base.ts";

export type SklearnLikeModel = {
  predictProba(features: unknown): number[][] | Promise<number[][]>;
};

export type VectorizerLike = {
  transform(texts: string[]): unknown;
};

export class SklearnClassifier implements Classifier {
  public labels: string[];
  private model: SklearnLikeModel;
  private vectorizer?: VectorizerLike;

  constructor(model: SklearnLikeModel, labels: string[], vectorizer?: VectorizerLike) {
    this.model = model;
    this.labels = labels;
    this.vectorizer = vectorizer;
  }

  async classify(text: string): Promise<ClassificationResult> {
    const features = this.vectorizer ? this.vectorizer.transform([text]) : [text];
    const probabilities = await this.model.predictProba(features);
    const row = probabilities[0] ?? [];
    if (row.length === 0) {
      throw new Error("predictProba returned no probabilities for input");
    }

    let bestIndex = 0;
    for (let i = 1; i < row.length; i += 1) {
      if (row[i]! > row[bestIndex]!) {
        bestIndex = i;
      }
    }

    return {
      label: this.labels[bestIndex] ?? String(bestIndex),
      confidence: Number(row[bestIndex] ?? 0),
      rawOutput: row,
    };
  }
}
