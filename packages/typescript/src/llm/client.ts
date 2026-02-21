export interface LLMClient {
  complete(
    model: string,
    systemPrompt: string,
    prompt: string,
    temperature?: number,
  ): Promise<{ content: string; tokens?: number; costUsd?: number }>;
}

export type LiteLLMClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
};

async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts = 3,
  baseDelayMs = 1000,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const isTransient =
        err instanceof TypeError ||
        (err instanceof Error && err.name === "AbortError") ||
        (err instanceof Error && /5\d{2}|429/.test(err.message));
      if (!isTransient || attempt === maxAttempts) {
        throw err;
      }
      const delay = baseDelayMs * Math.pow(2, attempt - 1);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastError;
}

export class LiteLLMClient implements LLMClient {
  private baseUrl: string;
  private apiKey: string | null;
  private timeoutMs: number;

  constructor(options: LiteLLMClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? process.env.LITELLM_BASE_URL ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(/\/$/, "");
    this.apiKey = options.apiKey ?? process.env.LITELLM_API_KEY ?? process.env.OPENAI_API_KEY ?? null;
    this.timeoutMs = options.timeoutMs ?? 60000;
  }

  async complete(
    model: string,
    systemPrompt: string,
    prompt: string,
    temperature = 0,
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
          throw new Error(`LLM request failed (${response.status}): ${detail}`);
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
    });
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
