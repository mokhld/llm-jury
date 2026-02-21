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
