import type { ClassificationResult, Classifier } from "./base.ts";

export type HuggingFaceLabelScore = {
  label: string;
  score: number;
};

export type HuggingFacePipeline = (
  text: string,
  options?: Record<string, unknown>,
) =>
  | HuggingFaceLabelScore[]
  | Promise<HuggingFaceLabelScore[]>
  | HuggingFaceLabelScore[][]
  | Promise<HuggingFaceLabelScore[][]>;

export type HuggingFaceClassifierOptions = {
  modelName?: string;
  device?: string;
  pipeline?: HuggingFacePipeline;
  /**
   * Labels the jury debates over. When omitted they are taken from the
   * model's full score list on the first call.
   */
  labels?: string[];
};

// transformers.js returns only the top label unless asked for every score.
// `topk` is the @xenova/transformers v2 option and `top_k` the
// @huggingface/transformers v3 one; null means "all labels" in both.
const ALL_SCORES_OPTIONS: Record<string, unknown> = { topk: null, top_k: null };

function scoreList(raw: unknown): HuggingFaceLabelScore[] {
  if (Array.isArray(raw)) {
    return (Array.isArray(raw[0]) ? raw[0] : raw) as HuggingFaceLabelScore[];
  }
  if (raw && typeof raw === "object" && "label" in raw) {
    return [raw as HuggingFaceLabelScore];
  }
  return [];
}

/**
 * Local transformers.js text-classification pipeline as the primary
 * classifier. Results report `costUsd: 0` because no paid API is called.
 */
export class HuggingFaceClassifier implements Classifier {
  public labels: string[];
  private modelName?: string;
  private device: string;
  private pipeline?: HuggingFacePipeline;

  constructor(options: HuggingFaceClassifierOptions = {}) {
    this.labels = options.labels ? [...options.labels] : [];
    this.modelName = options.modelName;
    this.device = options.device ?? "cpu";
    this.pipeline = options.pipeline;
  }

  async classify(text: string): Promise<ClassificationResult> {
    const runner = await this.resolvePipeline();
    const normalized = scoreList(await runner(text, { ...ALL_SCORES_OPTIONS }));
    if (normalized.length === 0) {
      throw new Error("HuggingFace pipeline returned no scores");
    }

    let top = normalized[0]!;
    for (const item of normalized.slice(1)) {
      if (item.score > top.score) {
        top = item;
      }
    }

    // A single score is not the full label distribution (e.g. an injected
    // pipeline that ignores the all-scores options), so it does not set labels.
    if (this.labels.length === 0 && normalized.length > 1) {
      this.labels = normalized.map((item) => item.label);
    }

    return {
      label: top.label,
      confidence: Number(top.score),
      rawOutput: normalized,
      costUsd: 0,
    };
  }

  private async resolvePipeline(): Promise<HuggingFacePipeline> {
    if (this.pipeline) {
      return this.pipeline;
    }

    if (!this.modelName) {
      throw new Error("Provide modelName or pipeline to HuggingFaceClassifier.");
    }

    try {
      const transformers = (await import("@xenova/transformers")) as {
        pipeline: (task: string, modelName: string, options?: Record<string, unknown>) => Promise<HuggingFacePipeline>;
      };
      this.pipeline = await transformers.pipeline("text-classification", this.modelName, {
        device: this.device,
      });
      return this.pipeline;
    } catch (error) {
      throw new Error(
        "Unable to initialize HuggingFace pipeline. Install @xenova/transformers or inject a pipeline.",
        { cause: error as Error },
      );
    }
  }
}
