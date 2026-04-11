"""
Research Pipeline

Orchestrates the four-stage research loop:
  Planner -> Searcher -> Synthesizer -> Verifier

Design decisions:
  - Planning is optional (skip_planning=True runs direct synthesis for ablation)
  - Verification is optional (skip_verification=True skips the grounding pass)
  - tool_calls is tracked end-to-end for efficiency comparison
  - Failure recovery is built into the Searcher; the Pipeline surfaces
    unresolved sub-questions explicitly in the final ResearchAnswer
"""

from __future__ import annotations

import os
from typing import Any

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = Any  # type: ignore[assignment]

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover
    TavilyClient = Any  # type: ignore[assignment]

from .models import ResearchAnswer, Evidence
from .planner import ResearchPlanner
from .searcher import ResearchSearcher
from .synthesizer import ResearchSynthesizer
from .verifier import ResearchVerifier


def _format_output(answer: ResearchAnswer) -> str:
    lines = []
    lines.append(answer.answer_text)
    lines.append("")
    lines.append("## Sources")
    for s in answer.sources:
        lines.append(f"{s['number']}. [{s['title']}]({s['url']})")

    if answer.unverified_claims:
        lines.append("")
        lines.append("## Unverified Claims")
        lines.append(
            "The following claims could not be traced to retrieved evidence:"
        )
        for c in answer.unverified_claims:
            lines.append(f"- {c}")

    if answer.unanswered_sub_questions:
        lines.append("")
        lines.append("## Coverage Gaps")
        lines.append(
            "The following aspects could not be answered from available sources:"
        )
        for sq in answer.unanswered_sub_questions:
            lines.append(f"- {sq}")

    return "\n".join(lines)


class ResearchPipeline:
    """End-to-end orchestrator for planning, retrieval, synthesis, and verification."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        tavily_api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        results_per_query: int = 4,
    ):
        anthropic_key = anthropic_api_key or os.environ["ANTHROPIC_API_KEY"]
        tavily_key = tavily_api_key or os.environ["TAVILY_API_KEY"]

        self.anthropic = Anthropic(api_key=anthropic_key)
        self.tavily = TavilyClient(api_key=tavily_key)
        self.model = model
        self.results_per_query = results_per_query

        self.planner = ResearchPlanner(self.anthropic, model)
        self.searcher = ResearchSearcher(
            self.anthropic, self.tavily, model, results_per_query
        )
        self.synthesizer = ResearchSynthesizer(self.anthropic, model)
        self.verifier = ResearchVerifier(self.anthropic, model)

    def run(
        self,
        question: str,
        skip_planning: bool = False,
        skip_verification: bool = False,
        verbose: bool = False,
    ) -> ResearchAnswer:
        """Run the full research pipeline and return structured output."""
        tool_calls = 0

        # Stage 1: Planning
        if skip_planning:
            sub_questions = [question]
            if verbose:
                print("[planner] skipped - using question directly")
        else:
            if verbose:
                print("[planner] decomposing question...")
            sub_questions, reasoning = self.planner.decompose(question)
            tool_calls += 1
            if verbose:
                print(f"[planner] {len(sub_questions)} sub-questions | {reasoning}")
                for i, sq in enumerate(sub_questions, 1):
                    print(f"  {i}. {sq}")

        # Stage 2: Search + semantic compression
        all_evidence: list[Evidence] = []
        for sq in sub_questions:
            if verbose:
                print(f"[searcher] searching: {sq}")
            evidence, calls = self.searcher.search(sq)
            tool_calls += calls
            all_evidence.extend(evidence)
            if verbose:
                good = sum(1 for e in evidence if e.search_successful)
                print(f"[searcher] {good} useful sources found ({calls} API calls)")

        # Stage 3: Synthesis
        if verbose:
            print("[synthesizer] synthesizing answer...")
        answer_text, unanswered_sqs, sources = self.synthesizer.synthesize(
            question, sub_questions, all_evidence
        )
        tool_calls += 1

        # Stage 4: Verification
        claims = []
        unverified_texts: list[str] = []

        if not skip_verification:
            if verbose:
                print("[verifier] checking claims against evidence...")
            claims, unverified_texts = self.verifier.verify(
                answer_text, all_evidence, sources
            )
            tool_calls += 2  # extract_claims + verify
            if verbose:
                verified_count = sum(1 for c in claims if c.verified)
                print(
                    f"[verifier] {verified_count}/{len(claims)} claims verified, "
                    f"{len(unverified_texts)} flagged"
                )

        return ResearchAnswer(
            question=question,
            sub_questions=sub_questions,
            evidence=all_evidence,
            answer_text=answer_text,
            sources=sources,
            claims=claims,
            unverified_claims=unverified_texts,
            unanswered_sub_questions=unanswered_sqs,
            tool_calls=tool_calls,
        )

    def run_formatted(
        self,
        question: str,
        skip_planning: bool = False,
        skip_verification: bool = False,
        verbose: bool = True,
    ) -> str:
        """Run the pipeline and return a formatted markdown result."""
        result = self.run(question, skip_planning, skip_verification, verbose)
        return _format_output(result)
