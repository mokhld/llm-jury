export type Persona = {
  name: string;
  role: string;
  systemPrompt: string;
  model: string;
  temperature: number;
  knownBias?: string;
};

export type PersonaResponse = {
  personaName: string;
  label: string;
  confidence: number;
  reasoning: string;
  keyFactors: string[];
  dissentNotes?: string;
  rawResponse?: string;
  tokensUsed?: number;
  costUsd?: number;
};
