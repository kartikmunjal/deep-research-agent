"""
Research Planner

Decomposes a broad research question into 3-5 targeted sub-questions,
each independently searchable. The decomposition improves both recall
(sub-questions surface different documents) and synthesis coherence
(each sub-question maps to a distinct section of the final answer).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic

from .models import QueryCost


DECOMPOSE_PROMPT = """You are a research planning assistant. Your job is to break a complex research question into focused sub-questions that together constitute a complete answer.

Rules:
- Generate between 3 and 5 sub-questions
- Each sub-question must be independently searchable via a web search engine
- Sub-questions should be complementary, not redundant
- Order them logically: background first, then specifics, then implications
- Keep each sub-question under 15 words

Respond with a JSON object in this exact format:
{{
  "sub_questions": [
    "Sub-question 1",
    "Sub-question 2",
    "Sub-question 3"
  ],
  "reasoning": "One sentence explaining the decomposition strategy"
}}

Research question: {question}"""


class ResearchPlanner:
    """Generates a search plan from an input research question."""

    def __init__(self, client: Anthropic, model: str):
        self.client = client
        self.model = model

    def decompose(self, question: str, cost: QueryCost | None = None) -> tuple[list[str], str]:
        """Return 3-5 focused sub-questions and brief planning rationale."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}
            ],
        )
        if cost is not None:
            cost.add_response(response.usage)

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present.
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        sub_questions = parsed["sub_questions"]
        reasoning = parsed.get("reasoning", "")

        if not isinstance(sub_questions, list) or not sub_questions:
            raise ValueError("Planner returned invalid sub_questions payload")

        return [str(sq) for sq in sub_questions], str(reasoning)
