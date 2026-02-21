export type LLMReply = {
  content: string;
  tokens?: number;
  costUsd?: number;
};

export class FakeLLMClient {
  public calls: Array<{ model: string; systemPrompt: string; prompt: string; temperature: number }> = [];
  private replies: Record<string, LLMReply>;

  constructor(replies: Record<string, LLMReply> = {}) {
    this.replies = replies;
  }

  async complete(model: string, systemPrompt: string, prompt: string, temperature = 0): Promise<{ content: string; tokens: number; costUsd: number }> {
    this.calls.push({ model, systemPrompt, prompt, temperature });

    if (this.replies[model]) {
      const reply = this.replies[model];
      return {
        content: reply.content,
        tokens: reply.tokens ?? 10,
        costUsd: reply.costUsd ?? 0.001,
      };
    }

    for (const [key, reply] of Object.entries(this.replies)) {
      if (prompt.includes(key) || systemPrompt.includes(key) || model === key) {
        return {
          content: reply.content,
          tokens: reply.tokens ?? 10,
          costUsd: reply.costUsd ?? 0.001,
        };
      }
    }

    return {
      content: JSON.stringify({
        label: "safe",
        confidence: 0.75,
        reasoning: "Default safe response",
        key_factors: ["default"],
      }),
      tokens: 10,
      costUsd: 0.001,
    };
  }
}
