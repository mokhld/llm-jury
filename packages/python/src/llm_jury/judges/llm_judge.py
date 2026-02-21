from __future__ import annotations

from llm_jury._defaults import DEFAULT_MODEL
from llm_jury.debate.engine import DebateTranscript
from llm_jury.llm.client import LLMClient, LiteLLMClient
from llm_jury.utils import clamp_confidence, safe_json_parse, strip_markdown_fences

from .base import JudgeStrategy, Verdict


class LLMJudge(JudgeStrategy):
    DEFAULT_SYSTEM_PROMPT = (
        "You are the presiding judge in an expert panel. "
        "You have received assessments from multiple domain experts on a classification task.\n\n"
        "Your role is to:\n"
        "1. Weigh each expert's reasoning on its merits\n"
        "2. Consider the strength of evidence each expert cites\n"
        "3. Note where experts agree and disagree\n"
        "4. Factor in each expert's known perspective/bias\n"
        "5. If a debate summary is provided, use it to identify the decisive arguments\n"
        "6. Deliver a final classification with clear reasoning\n\n"
        "Respond ONLY with valid JSON:\n"
        "{\n"
        '  "label": "<final classification>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "reasoning": "<your synthesis of the debate>",\n'
        '  "key_agreements": ["<points all experts agreed on>"],\n'
        '  "key_disagreements": ["<points of contention>"],\n'
        '  "decisive_factor": "<what tipped the decision>"\n'
        "}"
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.llm_client = llm_client or LiteLLMClient()

    async def judge(self, transcript: DebateTranscript, labels: list[str]) -> Verdict:
        prompt = self._build_prompt(transcript, labels)
        payload = await self.llm_client.complete(
            model=self.model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            temperature=self.temperature,
        )
        raw_content = str(payload.get("content", "{}"))
        data = safe_json_parse(strip_markdown_fences(raw_content))

        if data is None:
            return Verdict(
                label=transcript.primary_result.label,
                confidence=transcript.primary_result.confidence,
                reasoning="LLM judge response was not valid JSON. Falling back to primary result.",
                was_escalated=True,
                primary_result=transcript.primary_result,
                debate_transcript=transcript,
                judge_strategy="llm_judge_fallback_invalid_json",
                total_duration_ms=transcript.duration_ms,
                total_cost_usd=(transcript.total_cost_usd or 0.0) + float(payload.get("cost_usd", 0.0) or 0.0),
            )

        return Verdict(
            label=str(data.get("label", transcript.primary_result.label)),
            confidence=clamp_confidence(data.get("confidence", transcript.primary_result.confidence)),
            reasoning=str(data.get("reasoning", "LLM judge response.")),
            was_escalated=True,
            primary_result=transcript.primary_result,
            debate_transcript=transcript,
            judge_strategy="llm_judge",
            total_duration_ms=transcript.duration_ms,
            total_cost_usd=(transcript.total_cost_usd or 0.0) + float(payload.get("cost_usd", 0.0) or 0.0),
        )

    def _build_prompt(self, transcript, labels: list[str]) -> str:
        lines = [
            f"Input: {transcript.input_text}",
            f"Available labels: {', '.join(labels)}",
            f"Primary result: {transcript.primary_result.label} ({transcript.primary_result.confidence:.2f})",
        ]

        for round_idx, round_responses in enumerate(transcript.rounds):
            heading = "Initial Expert Opinions" if round_idx == 0 else f"Revised Opinions (Round {round_idx + 1})"
            lines.append(f"\n{heading}:")
            for response in round_responses:
                lines.append(
                    f"- {response.persona_name}: {response.label} ({response.confidence:.2f}) | "
                    f"Reasoning: {response.reasoning}"
                )

        if getattr(transcript, "summary", None):
            lines.append(f"\nDebate Summary:\n{transcript.summary}")

        lines.append(
            "\nRespond ONLY with JSON containing: "
            "label, confidence, reasoning, key_agreements, key_disagreements, decisive_factor."
        )
        return "\n".join(lines)
