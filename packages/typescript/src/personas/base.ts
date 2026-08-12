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
  // True when this response is a placeholder for a persona whose LLM call
  // failed or returned unparseable output. Failed responses stay in the
  // transcript for auditability but carry no vote: judges, consensus
  // checks, and prompt builders all skip them.
  failed?: boolean;
};
