import test from "node:test";
import assert from "node:assert/strict";

import { LLMClassifier } from "../../src/classifiers/llmClassifier.ts";
import { Jury } from "../../src/jury/core.ts";
import { LLMJudge } from "../../src/judges/llmJudge.ts";
import type { Persona } from "../../src/personas/base.ts";

test(
  "real api smoke classification pipeline",
  { skip: !process.env.OPENAI_API_KEY },
  async () => {
    const classifier = new LLMClassifier({
      model: "gpt-5-mini",
      labels: ["safe", "unsafe"],
      systemPrompt: "Classify text as safe or unsafe.",
    });

    const personas: Persona[] = [
      {
        name: "Policy Analyst",
        role: "Policy reasoning",
        systemPrompt: "Assess policy risk conservatively.",
        model: "gpt-5-mini",
        temperature: 0.3,
      },
      {
        name: "Context Expert",
        role: "Context reasoning",
        systemPrompt: "Assess context and non-literal usage.",
        model: "gpt-5-mini",
        temperature: 0.3,
      },
    ];

    const jury = new Jury({
      classifier,
      personas,
      confidenceThreshold: 1.01,
      judge: new LLMJudge({ model: "gpt-5-mini" }),
      maxDebateCostUsd: 2.0,
    });

    const verdict = await jury.classify("Hello, how are you?");
    assert.ok(["safe", "unsafe"].includes(verdict.label));
    assert.ok(verdict.confidence >= 0 && verdict.confidence <= 1);
    assert.equal(verdict.wasEscalated, true);
  },
);
