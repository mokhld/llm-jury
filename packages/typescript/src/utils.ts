export function stripMarkdown(content: string): string {
  const trimmed = content.trim();
  if (trimmed.startsWith("```")) {
    return trimmed
      .split("\n")
      .filter((line) => !line.trim().startsWith("```"))
      .join("\n")
      .trim();
  }
  return trimmed;
}

export function safeJsonObject(content: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(content) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}

export type Semaphore = {
  acquire: () => Promise<void>;
  release: () => void;
};

export function createSemaphore(permits: number): Semaphore {
  let available = Math.max(1, permits);
  const waiters: Array<() => void> = [];

  return {
    async acquire() {
      if (available > 0) {
        available -= 1;
        return;
      }
      await new Promise<void>((resolve) => waiters.push(resolve));
    },
    release() {
      const next = waiters.shift();
      if (next) {
        next();
      } else {
        available += 1;
      }
    },
  };
}
