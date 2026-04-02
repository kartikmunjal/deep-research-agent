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
