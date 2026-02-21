export type ClassificationResult = {
  label: string;
  confidence: number;
  rawOutput?: unknown;
};

export interface Classifier {
  labels: string[];
  classify(text: string): Promise<ClassificationResult>;
  classifyBatch?(texts: string[]): Promise<ClassificationResult[]>;
}

export async function defaultClassifyBatch(classifier: Classifier, texts: string[]): Promise<ClassificationResult[]> {
  const results: ClassificationResult[] = [];
  for (const text of texts) {
    results.push(await classifier.classify(text));
  }
  return results;
}
