# llm-jury (Python)

Confidence-based escalation middleware for classifier edge cases. When your classifier is uncertain, let a configurable jury of LLM personas debate and return an auditable verdict.

Default LLM model: `gpt-5-mini` (overridable per-component).

## Install

```bash
pip install llm-jury
```

## Quick Start

```python
import asyncio
from llm_jury import Jury, PersonaRegistry
from llm_jury.classifiers import LLMClassifier
from llm_jury.judges import MajorityVoteJudge

classifier = LLMClassifier(labels=["safe", "unsafe"])

jury = Jury(
    classifier=classifier,
    personas=PersonaRegistry.content_moderation(),
    confidence_threshold=0.85,
    judge=MajorityVoteJudge(),
)

async def main():
    verdict = await jury.classify("That group always causes problems")
    print(f"Label: {verdict.label}")
    print(f"Confidence: {verdict.confidence}")
    print(f"Escalated: {verdict.was_escalated}")
    print(f"Strategy: {verdict.judge_strategy}")

    if verdict.debate_transcript:
        for persona_resp in verdict.debate_transcript.rounds[-1]:
            print(f"  {persona_resp.persona_name}: {persona_resp.label} "
                  f"({persona_resp.confidence}) — {persona_resp.reasoning[:80]}")

    # Serialise for logging/storage
    print(verdict.to_json())

asyncio.run(main())
```

Set `OPENAI_API_KEY` in your environment before running.

## What You Get Back

`jury.classify(text)` returns a `Verdict`. There are two shapes depending on whether the input was escalated.

### Fast path (confidence above threshold)

When the primary classifier is confident enough, the verdict is returned directly — no debate.

```json
{
  "label": "safe",
  "confidence": 0.95,
  "reasoning": "Classified by primary classifier with sufficient confidence.",
  "was_escalated": false,
  "primary_result": {
    "label": "safe",
    "confidence": 0.95,
    "raw_output": { "label": "safe", "confidence": 0.95 }
  },
  "debate_transcript": null,
  "judge_strategy": "primary_classifier",
  "total_duration_ms": 312,
  "total_cost_usd": 0.0001
}
```

### Escalated (confidence below threshold)

When confidence is too low, the input goes through persona debate and a judge produces the final verdict.

```json
{
  "label": "unsafe",
  "confidence": 1.0,
  "reasoning": "The statement is a sweeping negative generalization about an entire group.",
  "was_escalated": true,
  "primary_result": {
    "label": "unsafe",
    "confidence": 0.62
  },
  "debate_transcript": {
    "input_text": "Those people always cause problems wherever they go",
    "rounds": [
      [
        {
          "persona_name": "Policy Analyst",
          "label": "unsafe",
          "confidence": 0.90,
          "reasoning": "Blanket negative generalization targeting a group.",
          "key_factors": ["group-targeting language"],
          "tokens_used": 185,
          "cost_usd": 0.0003
        },
        {
          "persona_name": "Cultural Context Expert",
          "label": "unsafe",
          "confidence": 0.85,
          "reasoning": "No mitigating context; phrasing is unambiguously negative.",
          "key_factors": ["derogatory framing"],
          "tokens_used": 192,
          "cost_usd": 0.0003
        },
        {
          "persona_name": "Harm Assessment Specialist",
          "label": "unsafe",
          "confidence": 0.92,
          "reasoning": "Risks normalizing prejudice against the targeted group.",
          "key_factors": ["potential for real-world harm"],
          "tokens_used": 178,
          "cost_usd": 0.0003
        }
      ]
    ],
    "summary": "The experts unanimously agreed the statement constitutes an unsafe sweeping generalization targeting a group.",
    "duration_ms": 2450,
    "total_tokens": 555,
    "total_cost_usd": 0.0009
  },
  "judge_strategy": "majority_vote",
  "total_duration_ms": 2780,
  "total_cost_usd": 0.001
}
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Final classification |
| `confidence` | `float` | Final confidence (0.0–1.0) |
| `reasoning` | `str` | Human-readable explanation |
| `was_escalated` | `bool` | Whether debate was triggered |
| `primary_result` | `ClassificationResult` | Fast-path classifier output (`label`, `confidence`, `raw_output`) |
| `debate_transcript` | `DebateTranscript \| None` | Full debate audit trail (`rounds`, `summary`, `duration_ms`, `total_tokens`, `total_cost_usd`) |
| `judge_strategy` | `str` | Strategy that produced the verdict |
| `total_duration_ms` | `int` | Wall-clock time (ms) |
| `total_cost_usd` | `float \| None` | API cost in USD |

`debate_transcript` also includes `summary` (`str | None`) — a structured summary produced during the Summarisation stage of the deliberation pipeline (null in non-deliberation modes).

Each persona response in `debate_transcript.rounds` contains: `persona_name`, `label`, `confidence`, `reasoning`, `key_factors`, `dissent_notes`, `tokens_used`, `cost_usd`.

## Choosing What to Pass In

### Classifiers

| Classifier | When to use | Example |
|---|---|---|
| `FunctionClassifier` | You already have a classifier and want to wrap it | `FunctionClassifier(fn=my_model, labels=["a","b"])` |
| `LLMClassifier` | Your primary classifier IS an LLM | `LLMClassifier(labels=["safe","unsafe"])` |
| `HuggingFaceClassifier` | Use a HuggingFace model locally | `HuggingFaceClassifier("unitary/toxic-bert")` |
| `SklearnClassifier` | Wrap an sklearn model | `SklearnClassifier(model, labels, vectorizer)` |

### Persona Sets

| Registry method | Domain | Personas included |
|---|---|---|
| `PersonaRegistry.content_moderation()` | Trust & Safety | Policy Analyst, Cultural Context Expert, Harm Assessment Specialist |
| `PersonaRegistry.legal_compliance()` | Legal/Regulatory | Regulatory Attorney, Business Risk Analyst, Industry Standards Expert |
| `PersonaRegistry.medical_triage()` | Healthcare | Clinical Safety Reviewer, Contextual Historian, Resource Allocation Analyst |
| `PersonaRegistry.financial_compliance()` | AML/KYC | AML Investigator, Risk Quant, Business Controls Reviewer |
| `PersonaRegistry.custom([...])` | Any domain | Build your own from dicts |

### Judge Strategies

| Strategy | How it decides | Best for |
|---|---|---|
| `MajorityVoteJudge()` | Counts votes from final round. Confidence = fraction agreeing. | Simple, fast, no extra LLM call |
| `WeightedVoteJudge()` | Weights votes by persona confidence scores. | When some personas are more certain than others |
| `LLMJudge()` | Reads the full transcript and synthesises a reasoned verdict. | Maximum reasoning quality, auditable synthesis |
| `BayesianJudge()` | Bayesian aggregation with optional per-persona priors. | When you have calibration data on persona reliability |

### Debate Modes

| Mode | Behaviour | Best for |
|---|---|---|
| `independent` | All personas assess in parallel, no cross-talk | Fast, low cost |
| `sequential` | Each persona sees all previous responses | When later opinions should build on earlier ones |
| `deliberation` (default) | Full 4-stage CEJ pipeline: Initial Opinions → Structured Debate → Summarisation → Final Judgment | Maximum value; complex edge cases where structured discussion helps |
| `adversarial` | Assigns prosecution/defense stances to alternate personas | Stress-testing a classification from both sides |

## Important Notes

- **Temperature is handled automatically.** The SDK omits the temperature parameter for reasoning models (`gpt-5*`, `o1*`, `o3*`). No configuration needed.
- **Escalation condition is strictly `< threshold`** — a confidence exactly equal to the threshold does NOT escalate.
- **All LLM calls have automatic retry** (3 attempts with exponential backoff) via tenacity.
- **Default debate mode is deliberation** for maximum value — it runs the full 4-stage CEJ pipeline. For cheaper operation, use `DebateConfig(mode=DebateMode.INDEPENDENT)`.
- **Cost tracking** — `total_cost_usd` on verdicts is estimated from token usage via litellm's model pricing table, not from provider billing. Accurate for known models; may be `None` for unrecognised ones.

## Examples

Runnable examples in `examples/` (require `OPENAI_API_KEY` unless noted):

```bash
python examples/content_moderation.py   # Content moderation with LLM classifier
python examples/custom_personas.py      # Custom persona definitions + deliberation mode
python examples/legal_compliance.py     # Legal compliance with sequential debate + weighted vote
python examples/threshold_calibration.py # Threshold calibration (no API key needed)
```

## Prerequisites

- Python `3.10+`
- `OPENAI_API_KEY` for real LLM calls
- `litellm` installed (included in dependencies)

## Running Tests

```bash
cd packages/python
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Smoke test with real API:

```bash
OPENAI_API_KEY="..." python -m pytest tests/test_integration/test_smoke_real_api.py -v
```

## CLI

The CLI is for batch workflows. The primary interface is the Python API above.

```bash
# Classify a JSONL file
llm-jury classify \
  --input input.jsonl \
  --output verdicts.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --threshold 0.7 \
  --labels safe,unsafe

# Calibrate threshold from labelled data
llm-jury calibrate \
  --input calibration.jsonl \
  --classifier function \
  --personas content_moderation \
  --judge majority \
  --labels safe,unsafe
```

Input JSONL format for `classify`:
```json
{"text": "some text", "predicted_label": "safe", "predicted_confidence": 0.95}
```

Input JSONL format for `calibrate` (requires ground-truth `label`):
```json
{"text": "some text", "label": "safe", "predicted_label": "safe", "predicted_confidence": 0.95}
```

Supported classifier specs: `function`, `llm:<model>`, `huggingface:<model>`.

## License

MIT
