import { NOOP_LOGGER } from "../logger.ts";
import type { Logger } from "../logger.ts";

export interface LLMClient {
  complete(
    model: string,
    systemPrompt: string,
    prompt: string,
    temperature?: number,
    responseFormat?: Record<string, unknown>,
  ): Promise<{ content: string; tokens?: number; costUsd?: number }>;
}

export type LiteLLMClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  logger?: Logger;
};

function readErrorStatus(err: unknown): number | undefined {
  if (!err || typeof err !== "object") return undefined;
  const e = err as Record<string, unknown>;
  for (const key of ["status", "statusCode", "code"] as const) {
    const value = e[key];
    if (typeof value === "number") return value;
  }
  const response = e.response as Record<string, unknown> | undefined;
  if (response && typeof response.status === "number") return response.status;
  return undefined;
}

export function isRetryableError(err: unknown): boolean {
  if (err instanceof TypeError) return true;
  if (err instanceof Error && err.name === "AbortError") return true;
  const status = readErrorStatus(err);
  if (typeof status === "number" && (status === 429 || (status >= 500 && status < 600))) {
    return true;
  }
  // Back-compat: errors thrown without a structured status but mentioning one in the message.
  if (err instanceof Error && /\b(?:429|5\d{2})\b/.test(err.message)) return true;
  return false;
}

async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts: number,
  baseDelayMs: number,
  logger: Logger,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (!isRetryableError(err) || attempt === maxAttempts) {
        throw err;
      }
      const delay = baseDelayMs * Math.pow(2, attempt - 1);
      logger.warn(`[llm-jury] LLM call failed (attempt ${attempt}/${maxAttempts}); retrying`, {
        delayMs: delay,
        error: err instanceof Error ? err.message : String(err),
      });
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastError;
}

export class LiteLLMClient implements LLMClient {
  private baseUrl: string;
  private apiKey: string | null;
  private timeoutMs: number;
  private logger: Logger;

  constructor(options: LiteLLMClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? process.env.LITELLM_BASE_URL ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(/\/$/, "");
    this.apiKey = options.apiKey ?? process.env.LITELLM_API_KEY ?? process.env.OPENAI_API_KEY ?? null;
    this.timeoutMs = options.timeoutMs ?? 60000;
    this.logger = options.logger ?? NOOP_LOGGER;
  }

  async complete(
    model: string,
    systemPrompt: string,
    prompt: string,
    temperature = 0,
    responseFormat?: Record<string, unknown>,
  ): Promise<{ content: string; tokens?: number; costUsd?: number }> {
    if (!this.apiKey) {
      throw new Error(
        "No API key configured. Set LITELLM_API_KEY or OPENAI_API_KEY, or inject a custom llmClient.",
      );
    }

    const body: {
      model: string;
      messages: Array<{ role: "system" | "user"; content: string }>;
      temperature?: number;
      response_format?: Record<string, unknown>;
    } = {
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
    };

    if (shouldSendTemperature(model, temperature)) {
      body.temperature = temperature;
    }
    if (responseFormat) {
      body.response_format = responseFormat;
    }

    return withRetry(async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const response = await fetch(`${this.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!response.ok) {
          const detail = await response.text();
          const httpError = new Error(`LLM request failed (${response.status}): ${detail}`) as Error & {
            status: number;
          };
          httpError.status = response.status;
          throw httpError;
        }

        const payload = (await response.json()) as {
          choices?: Array<{ message?: { content?: string } }>;
          usage?: { total_tokens?: number };
        };

        const content = payload.choices?.[0]?.message?.content;
        if (typeof content !== "string") {
          throw new Error("LLM response did not include choices[0].message.content");
        }

        return {
          content,
          tokens: Number(payload.usage?.total_tokens ?? 0),
          costUsd: undefined,
        };
      } finally {
        clearTimeout(timeout);
      }
    }, 3, 1000, this.logger);
  }
}

function shouldSendTemperature(model: string, temperature: number | undefined): boolean {
  if (typeof temperature !== "number" || Number.isNaN(temperature)) {
    return false;
  }
  const lower = model.toLowerCase();
  const noTempPrefixes = ["o1", "o3", "gpt-5"];
  return !noTempPrefixes.some((prefix) => lower.startsWith(prefix));
}
