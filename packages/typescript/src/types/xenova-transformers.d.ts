declare module "@xenova/transformers" {
  export function pipeline(
    task: string,
    modelName: string,
    options?: Record<string, unknown>,
  ): Promise<(text: string) => unknown>;
}
