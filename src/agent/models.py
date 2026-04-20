from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Evidence:
    url: str
    title: str
    extracted_text: str  # semantically compressed — only sentences relevant to sub_question
    sub_question: str
    search_successful: bool = True


@dataclass
class Claim:
    text: str
    citation_numbers: list[int]
    verified: bool
    supporting_excerpt: Optional[str] = None


@dataclass
class QueryCost:
    """Tracks token usage and API requests for one pipeline run.

    Pricing constants reflect claude-sonnet-4-6 public rates and
    Tavily Advanced search pricing as of 2026. Update if rates change.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    tavily_requests: int = 0

    # claude-sonnet-4-6: $3.00/M input, $15.00/M output
    _CLAUDE_INPUT_COST_PER_M: float = field(default=3.00, init=False, repr=False)
    _CLAUDE_OUTPUT_COST_PER_M: float = field(default=15.00, init=False, repr=False)
    # Tavily Advanced: ~$0.004/request
    _TAVILY_COST_PER_REQUEST: float = field(default=0.004, init=False, repr=False)

    def add_response(self, usage: object) -> None:
        """Accumulate token counts from an Anthropic response.usage object."""
        self.input_tokens += getattr(usage, "input_tokens", 0)
        self.output_tokens += getattr(usage, "output_tokens", 0)

    def add_tavily(self, n: int = 1) -> None:
        self.tavily_requests += n

    @property
    def estimate_usd(self) -> float:
        claude = (
            (self.input_tokens / 1_000_000) * self._CLAUDE_INPUT_COST_PER_M
            + (self.output_tokens / 1_000_000) * self._CLAUDE_OUTPUT_COST_PER_M
        )
        tavily = self.tavily_requests * self._TAVILY_COST_PER_REQUEST
        return round(claude + tavily, 4)


@dataclass
class ResearchAnswer:
    question: str
    sub_questions: list[str]
    evidence: list[Evidence]
    answer_text: str          # synthesized answer with inline [N] citations
    sources: list[dict]       # [{"number": 1, "title": ..., "url": ...}, ...]
    claims: list[Claim]
    unverified_claims: list[str]
    unanswered_sub_questions: list[str]
    tool_calls: int = 0
    cost: Optional[QueryCost] = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    sub_question_timings: list[dict] = field(default_factory=list)

    @property
    def hallucination_rate(self) -> float:
        if not self.claims:
            return 0.0
        return len(self.unverified_claims) / len(self.claims)

    @property
    def completeness(self) -> float:
        total = len(self.sub_questions)
        if total == 0:
            return 0.0
        answered = total - len(self.unanswered_sub_questions)
        return answered / total

    @property
    def citation_accuracy(self) -> float:
        if not self.claims:
            return 0.0
        verified = sum(1 for c in self.claims if c.verified)
        return verified / len(self.claims)
