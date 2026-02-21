import type { ClassificationResult, Classifier } from "./base.ts";

export type HuggingFaceLabelScore = {
  label: string;
  score: number;
};

export type HuggingFacePipeline = (
  text: string,
) =>
  | HuggingFaceLabelScore[]
  | Promise<HuggingFaceLabelScore[]>
  | HuggingFaceLabelScore[][]
  | Promise<HuggingFaceLabelScore[][]>;

export type HuggingFaceClassifierOptions = {
  modelName?: string;
  device?: string;
  pipeline?: HuggingFacePipeline;
};

export class HuggingFaceClassifier implements Classifier {
  public labels: string[];
  private modelName?: string;
  private device: string;
  private pipeline?: HuggingFacePipeline;

  constructor(options: HuggingFaceClassifierOptions = {}) {
    this.labels = [];
    this.modelName = options.modelName;
    this.device = options.device ?? "cpu";
    this.pipeline = options.pipeline;
  }

  async classify(text: string): Promise<ClassificationResult> {
    const runner = await this.resolvePipeline();
    const raw = await runner(text);
    const normalized = Array.isArray(raw[0]) ? (raw as HuggingFaceLabelScore[][])[0] : (raw as HuggingFaceLabelScore[]);
    if (!normalized || normalized.length === 0) {
      throw new Error("HuggingFace pipeline returned no scores");
    }

    let top = normalized[0]!;
    for (const item of normalized.slice(1)) {
      if (item.score > top.score) {
        top = item;
      }
    }

    if (this.labels.length === 0) {
      this.labels = normalized.map((item) => item.label);
    }

    return {
      label: top.label,
      confidence: Number(top.score),
      rawOutput: normalized,
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
