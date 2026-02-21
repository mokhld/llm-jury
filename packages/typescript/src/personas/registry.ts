import type { Persona } from "./base.ts";

function persona(p: Omit<Persona, "model" | "temperature"> & Partial<Pick<Persona, "model" | "temperature">>): Persona {
  return {
    model: "gpt-5-mini",
    temperature: 0.3,
    ...p,
  };
}

export class PersonaRegistry {
  static contentModeration(): Persona[] {
    return [
      persona({
        name: "Policy Analyst",
        role: "Interprets content against platform policies",
        systemPrompt:
          "You are a content policy analyst. Focus on explicit policy violations, edge cases in policy language, and precedent from similar decisions.",
        knownBias: "policy-strict",
      }),
      persona({
        name: "Cultural Context Expert",
        role: "Considers cultural and contextual nuances",
        systemPrompt:
          "You are an expert in cross-cultural communication and context. Consider satire, community norms, and linguistic register.",
        knownBias: "tends permissive on context",
      }),
      persona({
        name: "Harm Assessment Specialist",
        role: "Evaluates potential real-world impact",
        systemPrompt: "You evaluate potential real-world harm and impact on vulnerable groups.",
        knownBias: "harm-focused",
      }),
    ];
  }

  static legalCompliance(): Persona[] {
    return [
      persona({
        name: "Regulatory Attorney",
        role: "Strict legal interpretation",
        systemPrompt: "You are a regulatory compliance attorney.",
      }),
      persona({
        name: "Business Risk Analyst",
        role: "Weighs legal risk against business impact",
        systemPrompt: "You are a business risk analyst.",
      }),
      persona({
        name: "Industry Standards Expert",
        role: "Compares against industry norms",
        systemPrompt: "You are an expert in industry standards.",
      }),
    ];
  }

  static medicalTriage(): Persona[] {
    return [
      persona({
        name: "Clinical Safety Reviewer",
        role: "Prioritizes patient safety and urgency",
        systemPrompt: "You are a clinician focused on triage safety.",
      }),
      persona({
        name: "Contextual Historian",
        role: "Assesses relevant clinical context",
        systemPrompt: "You evaluate relevant context and confounders.",
      }),
      persona({
        name: "Resource Allocation Analyst",
        role: "Balances triage severity and capacity",
        systemPrompt: "You assess triage decisions against capacity constraints.",
      }),
    ];
  }

  static financialCompliance(): Persona[] {
    return [
      persona({ name: "AML Investigator", role: "Flags suspicious behavior patterns", systemPrompt: "You are an AML investigator." }),
      persona({ name: "Risk Quant", role: "Assesses probabilistic financial risk", systemPrompt: "You are a quantitative risk analyst." }),
      persona({
        name: "Business Controls Reviewer",
        role: "Assesses control proportionality",
        systemPrompt: "You assess control design and business practicality.",
      }),
    ];
  }

  static custom(personas: Persona[]): Persona[] {
    return personas.map((entry) => persona(entry));
  }
}
