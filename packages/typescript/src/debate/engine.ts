import type { ClassificationResult } from "../classifiers/base.ts";
import { LiteLLMClient } from "../llm/client.ts";
import type { LLMClient } from "../llm/client.ts";
import type { Persona, PersonaResponse } from "../personas/base.ts";
import { stripMarkdown } from "../utils.ts";

const SUMMARISATION_PROMPT =
  "You are a neutral summarisation agent. You have observed a structured debate " +
  "among domain experts about classifying a piece of text.\n\n" +
  "Produce a concise synthesis that covers:\n" +
  "1. The main arguments from each side\n" +
  "2. Points of consensus among the experts\n" +
  "3. Unresolved disagreements\n\n" +
  "Be factual and impartial. Do not add your own classification.";

const DELIBERATION_INSTRUCTIONS =
  "You have seen the initial assessments from all experts on this input. " +
  "You MUST:\n" +
  "(i) Engage with at least one other expert's reasoning — agree or disagree " +
  "with supporting rationale.\n" +
  "(ii) Revise your own classification if you find their counter-arguments compelling.\n" +
  "(iii) Re-evaluate the input through the interpretive lens of at least one other expert's " +
  "perspective, considering both intent and impact.\n\n" +
  "Then provide your revised assessment.";

export const DebateMode = {
  INDEPENDENT: "independent",
  SEQUENTIAL: "sequential",
  DELIBERATION: "deliberation",
  ADVERSARIAL: "adversarial",
} as const;

export type DebateMode = (typeof DebateMode)[keyof typeof DebateMode];

export type DebateTranscript = {
  inputText: string;
  primaryResult: ClassificationResult;
  rounds: PersonaResponse[][];
  summary?: string;
  durationMs: number;
  totalTokens: number;
  totalCostUsd: number | null;
};

export class DebateConfig {
  mode: DebateMode;
  maxRounds: number;
  includePrimaryResult: boolean;
  includeConfidence: boolean;

  constructor(options: Partial<DebateConfig> = {}) {
    this.mode = options.mode ?? DebateMode.DELIBERATION;
    this.maxRounds = options.maxRounds ?? 2;
    this.includePrimaryResult = options.includePrimaryResult ?? true;
    this.includeConfidence = options.includeConfidence ?? true;
  }
}

export class DebateEngine {
  private personas: Persona[];
  private config: DebateConfig;
  private llmClient: LLMClient;
  private concurrency: number;

  constructor(
    personas: Persona[],
    config = new DebateConfig(),
    llmClient: LLMClient = new LiteLLMClient(),
    concurrency = 5,
  ) {
    this.personas = personas;
    this.config = config;
    this.llmClient = llmClient;
    this.concurrency = Math.max(1, concurrency);
  }

  static jsonResponseBlock(): string {
    return (
      "\n## Your Assessment\n\n" +
      "Provide your classification. Respond ONLY with valid JSON:\n" +
      "```json\n" +
      "{\n" +
      '  "label": "<your classification>",\n' +
      '  "confidence": <0.0-1.0>,\n' +
      '  "reasoning": "<your full reasoning>",\n' +
      '  "key_factors": ["<factor 1>", "<factor 2>"],\n' +
      '  "dissent_notes": "<optional rebuttal against opposing side>"\n' +
      "}\n" +
      "```"
    );
  }

  async debate(
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    maxCostUsd: number | null = null,
  ): Promise<DebateTranscript> {
    const start = Date.now();
    const rounds: PersonaResponse[][] = [];
    let totalTokens = 0;
    let totalCostUsd = 0;

    if (this.personas.length === 0) {
      return {
        inputText: text,
        primaryResult,
        rounds: [],
        durationMs: Date.now() - start,
        totalTokens: 0,
        totalCostUsd: 0,
      };
    }

    if (this.config.mode === DebateMode.INDEPENDENT || this.config.mode === DebateMode.ADVERSARIAL) {
      const responses = await this.runRound(text, primaryResult, labels, []);
      rounds.push(responses);
      responses.forEach((response) => {
        totalTokens += Number(response.tokensUsed ?? 0);
        totalCostUsd += Number(response.costUsd ?? 0);
      });
    } else if (this.config.mode === DebateMode.SEQUENTIAL) {
      const responses: PersonaResponse[] = [];
      for (const persona of this.personas) {
        const response = await this.queryPersona(
          persona,
          text,
          primaryResult,
          labels,
          responses.length > 0 ? [responses] : [],
        );
        responses.push(response);
        totalTokens += Number(response.tokensUsed ?? 0);
        totalCostUsd += Number(response.costUsd ?? 0);
        if (maxCostUsd != null && totalCostUsd > maxCostUsd) {
          break;
        }
      }
      rounds.push(responses);
    } else if (this.config.mode === DebateMode.DELIBERATION) {
      const firstRound = await this.runRound(text, primaryResult, labels, []);
      rounds.push(firstRound);
      firstRound.forEach((response) => {
        totalTokens += Number(response.tokensUsed ?? 0);
        totalCostUsd += Number(response.costUsd ?? 0);
      });

      if (maxCostUsd != null && totalCostUsd > maxCostUsd) {
        return {
          inputText: text,
          primaryResult,
          rounds,
          durationMs: Date.now() - start,
          totalTokens,
          totalCostUsd,
        };
      }

      for (let i = 1; i < Math.max(1, this.config.maxRounds); i += 1) {
        const current = await this.runDeliberationRound(text, primaryResult, labels, rounds);
        rounds.push(current);
        current.forEach((response) => {
          totalTokens += Number(response.tokensUsed ?? 0);
          totalCostUsd += Number(response.costUsd ?? 0);
        });

        if (maxCostUsd != null && totalCostUsd > maxCostUsd) {
          break;
        }
        if (this.consensusReached(current)) {
          break;
        }
      }

      let summary: string | undefined;
      if (maxCostUsd == null || totalCostUsd <= maxCostUsd) {
        const summaryResult = await this.summarise(text, labels, rounds);
        totalTokens += summaryResult.tokens;
        totalCostUsd += summaryResult.cost;
        summary = summaryResult.summary;
      }

      return {
        inputText: text,
        primaryResult,
        rounds,
        summary,
        durationMs: Date.now() - start,
        totalTokens,
        totalCostUsd,
      };
    }

    return {
      inputText: text,
      primaryResult,
      rounds,
      durationMs: Date.now() - start,
      totalTokens,
      totalCostUsd,
    };
  }

  async runRound(
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): Promise<PersonaResponse[]> {
    const out: PersonaResponse[] = [];
    for (let i = 0; i < this.personas.length; i += this.concurrency) {
      const batch = this.personas.slice(i, i + this.concurrency);
      const responses = await Promise.all(
        batch.map((persona) => this.queryPersona(persona, text, primaryResult, labels, priorRounds)),
      );
      out.push(...responses);
    }
    return out;
  }

  async runDeliberationRound(
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): Promise<PersonaResponse[]> {
    const out: PersonaResponse[] = [];
    for (let i = 0; i < this.personas.length; i += this.concurrency) {
      const batch = this.personas.slice(i, i + this.concurrency);
      const responses = await Promise.all(
        batch.map((persona) =>
          this.queryPersonaDeliberation(persona, text, primaryResult, labels, priorRounds),
        ),
      );
      out.push(...responses);
    }
    return out;
  }

  async queryPersona(
    persona: Persona,
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): Promise<PersonaResponse> {
    const prompt = this.buildPersonaPrompt(persona, text, primaryResult, labels, priorRounds);
    const payload = await this.llmClient.complete(persona.model, persona.systemPrompt, prompt, persona.temperature);
    const parsed = this.parsePersonaResponse(payload.content, persona.name);
    parsed.rawResponse = payload.content;
    parsed.tokensUsed = Number(payload.tokens ?? 0);
    parsed.costUsd = Number(payload.costUsd ?? 0);
    return parsed;
  }

  async queryPersonaDeliberation(
    persona: Persona,
    text: string,
    primaryResult: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): Promise<PersonaResponse> {
    const prompt = this.buildDeliberationPrompt(persona, text, primaryResult, labels, priorRounds);
    const payload = await this.llmClient.complete(persona.model, persona.systemPrompt, prompt, persona.temperature);
    const parsed = this.parsePersonaResponse(payload.content, persona.name);
    parsed.rawResponse = payload.content;
    parsed.tokensUsed = Number(payload.tokens ?? 0);
    parsed.costUsd = Number(payload.costUsd ?? 0);
    return parsed;
  }

  buildPersonaPrompt(
    persona: Persona,
    text: string,
    primary: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): string {
    const parts: string[] = [];
    parts.push(`## Persona\n\n${persona.name}: ${persona.role}\n`);
    parts.push(`## Input to Classify\n\n${text}\n`);
    parts.push(`## Available Labels\n\n${labels.join(", ")}\n`);

    if (this.config.mode === DebateMode.ADVERSARIAL) {
      const personaIndex = this.personas.indexOf(persona);
      const stance = personaIndex % 2 === 0 ? "Prosecution" : "Defense";
      parts.push(
        "## Adversarial Role\n\n" +
          `You are assigned the **${stance}** side. Argue this stance rigorously while staying truthful to the evidence.\n`,
      );
    }

    if (this.config.includePrimaryResult) {
      const confidence = this.config.includeConfidence ? ` (confidence: ${primary.confidence.toFixed(2)})` : "";
      parts.push(
        "## Primary Classifier Result\n\n" +
          `Label: ${primary.label}${confidence}\n` +
          "Note: This was flagged as low-confidence and escalated to you.\n",
      );
    }

    if (priorRounds.length > 0) {
      parts.push("## Previous Assessments\n");
      priorRounds.forEach((roundResponses, idx) => {
        parts.push(`\n### Round ${idx + 1}\n`);
        for (const response of roundResponses) {
          parts.push(
            `**${response.personaName}**: ${response.label} (confidence: ${response.confidence.toFixed(2)})\n` +
              `Reasoning: ${response.reasoning}\n`,
          );
        }
      });
    }

    parts.push(DebateEngine.jsonResponseBlock());

    return parts.join("\n");
  }

  buildDeliberationPrompt(
    persona: Persona,
    text: string,
    primary: ClassificationResult,
    labels: string[],
    priorRounds: PersonaResponse[][],
  ): string {
    const parts: string[] = [];
    parts.push(`## Persona\n\n${persona.name}: ${persona.role}\n`);
    parts.push(`## Input to Classify\n\n${text}\n`);
    parts.push(`## Available Labels\n\n${labels.join(", ")}\n`);

    if (this.config.includePrimaryResult) {
      const confidence = this.config.includeConfidence ? ` (confidence: ${primary.confidence.toFixed(2)})` : "";
      parts.push(
        "## Primary Classifier Result\n\n" +
          `Label: ${primary.label}${confidence}\n` +
          "Note: This was flagged as low-confidence and escalated to you.\n",
      );
    }

    if (priorRounds.length > 0) {
      priorRounds.forEach((roundResponses, idx) => {
        if (idx === 0) {
          parts.push("## Initial Expert Opinions\n");
        } else {
          parts.push(`## Revised Opinions (Round ${idx + 1})\n`);
        }
        for (const response of roundResponses) {
          parts.push(
            `**${response.personaName}**: ${response.label} (confidence: ${response.confidence.toFixed(2)})\n` +
              `Reasoning: ${response.reasoning}\n`,
          );
        }
      });
    }

    parts.push(`\n## Deliberation Instructions\n\n${DELIBERATION_INSTRUCTIONS}\n`);

    parts.push(DebateEngine.jsonResponseBlock());

    return parts.join("\n");
  }

  async summarise(
    text: string,
    labels: string[],
    rounds: PersonaResponse[][],
  ): Promise<{ summary: string; tokens: number; cost: number }> {
    const parts: string[] = [];
    parts.push(`## Input\n\n${text}\n`);
    parts.push(`## Available Labels\n\n${labels.join(", ")}\n`);
    parts.push("## Expert Debate\n");
    rounds.forEach((roundResponses, idx) => {
      if (idx === 0) {
        parts.push("\n### Initial Expert Opinions\n");
      } else {
        parts.push(`\n### Revised Opinions (Round ${idx + 1})\n`);
      }
      for (const response of roundResponses) {
        parts.push(
          `**${response.personaName}**: ${response.label} (confidence: ${response.confidence.toFixed(2)})\n` +
            `Reasoning: ${response.reasoning}\n`,
        );
      }
    });

    const model = this.personas[0]?.model ?? "gpt-5-mini";
    const payload = await this.llmClient.complete(model, SUMMARISATION_PROMPT, parts.join("\n"), 0);

    return {
      summary: payload.content,
      tokens: Number(payload.tokens ?? 0),
      cost: Number(payload.costUsd ?? 0),
    };
  }

  parsePersonaResponse(raw: string, personaName: string): PersonaResponse {
    let parsed: {
      label?: string;
      confidence?: number;
      reasoning?: string;
      key_factors?: string[];
      dissent_notes?: string;
    };
    try {
      const candidate = JSON.parse(stripMarkdown(raw)) as unknown;
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
        throw new Error("Persona response must be a JSON object.");
      }
      parsed = candidate as {
        label?: string;
        confidence?: number;
        reasoning?: string;
        key_factors?: string[];
        dissent_notes?: string;
      };
    } catch {
      return {
        personaName,
        label: "unknown",
        confidence: 0,
        reasoning: `Failed to parse persona response: ${raw.slice(0, 200)}`,
        keyFactors: [],
      };
    }

    return {
      personaName,
      label: String(parsed.label ?? "unknown"),
      confidence: Number(parsed.confidence ?? 0),
      reasoning: String(parsed.reasoning ?? ""),
      keyFactors: Array.isArray(parsed.key_factors) ? parsed.key_factors.map(String) : [],
      dissentNotes: parsed.dissent_notes,
      tokensUsed: 0,
      costUsd: 0,
    };
  }

  consensusReached(roundResponses: PersonaResponse[]): boolean {
    if (roundResponses.length === 0) {
      return false;
    }
    const labels = new Set(roundResponses.map((r) => r.label));
    return labels.size === 1;
  }
}
