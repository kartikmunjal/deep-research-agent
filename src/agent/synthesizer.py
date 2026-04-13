"""
Research Synthesizer

Assembles retrieved evidence into a structured answer with inline citations.
The synthesizer receives semantically compressed evidence (not raw articles),
which keeps the prompt well within the context window even for broad questions
with many sub-questions and sources.

Citation format: [N] inline, with a numbered Sources list at the end.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic

from .models import Evidence


SYNTHESIZE_PROMPT = """You are a research analyst writing a comprehensive, cited answer.

Original question: {question}

Sub-questions to address:
{sub_questions_block}

Evidence (numbered for citation):
{evidence_block}

Instructions:
- Write a structured answer that addresses all sub-questions
- Use inline citations [N] wherever you make a factual claim
- Every factual claim must be supported by at least one citation
- If sources contradict each other, present both positions with their respective
  citations rather than picking one — explicitly flag the conflict and explain
  why the evidence disagrees (different study designs, time periods, methodologies)
- If a sub-question could not be answered due to missing evidence, state that explicitly
- Do NOT invent facts or extrapolate beyond what the evidence supports
- Be concise but complete — aim for 300-500 words

Format your response as valid JSON:
{{
  "answer": "Full answer text with [N] citations inline",
  "unanswered_sub_questions": ["sub-question text if truly unanswerable", ...]
}}"""


class ResearchSynthesizer:
    """Builds a cited answer from semantically compressed evidence."""

    def __init__(self, client: Anthropic, model: str):
        self.client = client
        self.model = model

    def synthesize(
        self,
        question: str,
        sub_questions: list[str],
        evidence: list[Evidence],
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        """Return answer text, unanswered sub-questions, and source metadata."""
        good_evidence = [e for e in evidence if e.search_successful and e.extracted_text]

        # Build numbered source list (deduplicated by URL).
        seen_urls: dict[str, int] = {}
        sources: list[dict[str, object]] = []
        numbered_evidence: list[tuple[int, Evidence]] = []

        for ev in good_evidence:
            if ev.url not in seen_urls:
                num = len(sources) + 1
                seen_urls[ev.url] = num
                sources.append({"number": num, "title": ev.title, "url": ev.url})
            numbered_evidence.append((seen_urls[ev.url], ev))

        # Identify sub-questions with no evidence.
        covered_sq = {ev.sub_question for ev in good_evidence}
        failed_sq = [sq for sq in sub_questions if sq not in covered_sq]

        # Build evidence block for the prompt.
        evidence_block_lines = []
        for num, ev in numbered_evidence:
            evidence_block_lines.append(f"[{num}] {ev.title} ({ev.url})")
            evidence_block_lines.append(f"    Sub-question: {ev.sub_question}")
            evidence_block_lines.append(f"    Extract: {ev.extracted_text}")
            evidence_block_lines.append("")

        sub_questions_block = "\n".join(f"{i+1}. {sq}" for i, sq in enumerate(sub_questions))

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": SYNTHESIZE_PROMPT.format(
                        question=question,
                        sub_questions_block=sub_questions_block,
                        evidence_block="\n".join(evidence_block_lines),
                    ),
                }
            ],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        answer_text = parsed["answer"]
        unanswered = parsed.get("unanswered_sub_questions", []) + failed_sq

        return answer_text, list(set(unanswered)), sources
