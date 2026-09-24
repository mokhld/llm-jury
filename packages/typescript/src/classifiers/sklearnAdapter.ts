import type { ClassificationResult, Classifier } from "./base.ts";

export type SklearnLikeModel = {
  predictProba(features: unknown): number[][] | Promise<number[][]>;
  /**
   * Class of each predictProba column, in column order (scikit-learn's
   * `classes_`). When present, it decides which label each column maps to.
   */
  classes?: ArrayLike<unknown>;
};

export type VectorizerLike = {
  transform(texts: string[]): unknown;
};

/**
 * Label for each predictProba column. When the model exposes `classes` and its
 * entries are the same set of strings as `labels`, columns are named by class.
 * Otherwise `labels` names the columns by position, which lets `labels` give
 * display names to non-string classes.
 */
function columnLabels(model: SklearnLikeModel, labels: string[]): string[] {
  if (model.classes == null) {
    return [...labels];
  }
  const classNames = Array.from(model.classes, (entry) => String(entry));
  if (classNames.length !== labels.length) {
    throw new Error(
      `model.classes has ${classNames.length} entries (${classNames.join(", ")}) but ` +
        `${labels.length} labels were given (${labels.join(", ")}). ` +
        "Pass one label per predictProba column, in model.classes order.",
    );
  }
  const classSet = new Set(classNames);
  const sameSet =
    classSet.size === classNames.length &&
    new Set(labels).size === labels.length &&
    labels.every((label) => classSet.has(label));
  return sameSet ? classNames : [...labels];
}

/**
 * scikit-learn style model with `predictProba` as the primary classifier.
 * Results report `costUsd: 0` because no paid API is called.
 */
export class SklearnClassifier implements Classifier {
  public labels: string[];
  private model: SklearnLikeModel;
  private vectorizer?: VectorizerLike;
  private columnLabels: string[];

  constructor(model: SklearnLikeModel, labels: string[], vectorizer?: VectorizerLike) {
    this.model = model;
    this.labels = labels;
    this.vectorizer = vectorizer;
    this.columnLabels = columnLabels(model, labels);
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
      label: this.columnLabels[bestIndex] ?? String(bestIndex),
      confidence: Number(row[bestIndex] ?? 0),
      rawOutput: row,
      costUsd: 0,
    };
  }
}
