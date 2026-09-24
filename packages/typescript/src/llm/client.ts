import { NOOP_LOGGER } from "../logger.ts";
import type { Logger } from "../logger.ts";

export interface LLMClient {
  complete(
    model: string,
    systemPrompt: string,
    prompt: string,
    temperature?: number,
    responseFormat?: Record<string, unknown>,
  ): Promise<{ content: string; tokens?: number; costUsd?: number; cached?: boolean }>;
}

export type LiteLLMClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  // Total attempts per call, including the first (default 3, minimum 1).
  // Only 429, 5xx, network errors and timeouts are retried.
  maxAttempts?: number;
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
  if (typeof status === "number") {
    return status === 429 || (status >= 500 && status < 600);
  }
  // Back-compat: errors thrown without a structured status but mentioning
  // one in the message. Only consulted when there is no structured status,
  // so a 400 whose body happens to contain "512" is not retried.
  if (err instanceof Error && /\b(?:429|5\d{2})\b/.test(err.message)) return true;
  return false;
}

const MAX_RETRY_AFTER_MS = 60_000;

function readHeader(headers: unknown, name: string): string | undefined {
  if (!headers || typeof headers !== "object") return undefined;
  const getter = (headers as { get?: unknown }).get;
  if (typeof getter === "function") {
    const value: unknown = getter.call(headers, name);
    return value == null ? undefined : String(value);
  }
  for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
    if (key.toLowerCase() === name && value != null) return String(value);
  }
  return undefined;
}

/**
 * Delay requested by a `Retry-After` header on the error (`err.headers` or
 * `err.response.headers`), in ms and capped at 60 s. Accepts delta-seconds
 * or an HTTP-date. Undefined when the header is absent or unparseable.
 */
function retryAfterMs(err: unknown): number | undefined {
  if (!err || typeof err !== "object") return undefined;
  const e = err as Record<string, unknown>;
  const response = e.response as Record<string, unknown> | undefined;
  const raw = readHeader(e.headers, "retry-after") ?? readHeader(response?.headers, "retry-after");
  if (raw === undefined) return undefined;
  const value = raw.trim();
  let delayMs: number;
  if (/^\d+(?:\.\d+)?$/.test(value)) {
    delayMs = Number(value) * 1000;
  } else {
    const at = Date.parse(value);
    if (Number.isNaN(at)) return undefined;
    delayMs = Math.max(0, at - Date.now());
  }
  return Math.min(MAX_RETRY_AFTER_MS, delayMs);
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
      const delay = retryAfterMs(err) ?? baseDelayMs * Math.pow(2, attempt - 1);
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
  private maxAttempts: number;
  private logger: Logger;

  constructor(options: LiteLLMClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? process.env.LITELLM_BASE_URL ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(/\/$/, "");
    this.apiKey = options.apiKey ?? process.env.LITELLM_API_KEY ?? process.env.OPENAI_API_KEY ?? null;
    this.timeoutMs = options.timeoutMs ?? 60000;
    this.maxAttempts = Math.max(1, Math.floor(options.maxAttempts ?? 3) || 1);
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
            headers: Headers;
          };
          httpError.status = response.status;
          httpError.headers = response.headers;
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
    }, this.maxAttempts, 1000, this.logger);
  }
}

function shouldSendTemperature(model: string, temperature: number | undefined): boolean {
  if (typeof temperature !== "number" || Number.isNaN(temperature)) {
    return false;
  }
  // Reasoning models reject a custom temperature. Match on the model name
  // after any provider prefix, so "openai/gpt-5-mini" is treated like
  // "gpt-5-mini".
  const lower = model.toLowerCase();
  const name = lower.slice(lower.lastIndexOf("/") + 1);
  const noTempPrefixes = ["o1", "o3", "gpt-5"];
  return !noTempPrefixes.some((prefix) => name.startsWith(prefix));
}
