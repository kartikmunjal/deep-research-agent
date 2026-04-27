"""
Research Pipeline

Orchestrates the four-stage research loop:
  Planner -> Searcher -> Synthesizer -> Verifier

Key design decisions:
  - run_async() parallelises all sub-question searches with asyncio.gather,
    reducing Stage 2 latency by ~N_subquestions (typically 3-5x)
  - run() is a sync wrapper around run_async() for scripts and eval harness
  - cost tracking via QueryCost accumulates token usage across every Claude
    call and Tavily request; cost.estimate_usd gives a per-query price
  - planning / verification are independently skippable for ablation
  - failure recovery (query reformulation, coverage gap surfacing) is in
    the Searcher; the Pipeline exposes unresolved sub-questions in ResearchAnswer
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from tavily import TavilyClient, AsyncTavilyClient

from .models import ResearchAnswer, Evidence, QueryCost
from .planner import ResearchPlanner
from .searcher import ResearchSearcher
from .synthesizer import ResearchSynthesizer
from .verifier import ResearchVerifier


def _format_output(answer: ResearchAnswer) -> str:
    lines = [answer.answer_text, "", "## Sources"]
    for s in answer.sources:
        lines.append(f"{s['number']}. [{s['title']}]({s['url']})")

    if answer.unverified_claims:
        lines += ["", "## Unverified Claims",
                  "The following claims could not be traced to retrieved evidence:"]
        for c in answer.unverified_claims:
            lines.append(f"- {c}")

    if answer.unanswered_sub_questions:
        lines += ["", "## Coverage Gaps",
                  "The following aspects could not be answered from available sources:"]
        for sq in answer.unanswered_sub_questions:
            lines.append(f"- {sq}")

    if answer.cost:
        lines += ["", f"*Est. cost: ${answer.cost.estimate_usd:.4f} | "
                  f"{answer.cost.input_tokens:,} in / {answer.cost.output_tokens:,} out tokens | "
                  f"{answer.cost.tavily_requests} searches*"]

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
        from anthropic import Anthropic, AsyncAnthropic
        from tavily import TavilyClient, AsyncTavilyClient

        anthropic_key = anthropic_api_key or os.environ["ANTHROPIC_API_KEY"]
        tavily_key = tavily_api_key or os.environ["TAVILY_API_KEY"]

        # Sync clients (planner, synthesizer, verifier; sync fallback for searcher)
        self._anthropic = Anthropic(api_key=anthropic_key)
        self._tavily = TavilyClient(api_key=tavily_key)
        # Async clients for parallel search
        self._async_anthropic = AsyncAnthropic(api_key=anthropic_key)
        self._async_tavily = AsyncTavilyClient(api_key=tavily_key)

        self.model = model

        self.planner = ResearchPlanner(self._anthropic, model)
        self.searcher = ResearchSearcher(
            self._anthropic, self._tavily, model, results_per_query,
            async_anthropic_client=self._async_anthropic,
            async_tavily_client=self._async_tavily,
        )
        self.synthesizer = ResearchSynthesizer(self._anthropic, model)
        self.verifier = ResearchVerifier(self._anthropic, model)

    # ------------------------------------------------------------------
    # Async-first implementation
    # ------------------------------------------------------------------

    async def run_async(
        self,
        question: str,
        skip_planning: bool = False,
        skip_verification: bool = False,
        verbose: bool = False,
    ) -> ResearchAnswer:
        """
        Run the research pipeline asynchronously.

        Stage 2 (search) fans out all sub-questions in parallel with
        asyncio.gather. For a typical 4-sub-question decomposition this
        reduces search latency by ~3-4x vs. serial execution.
        """
        cost = QueryCost()
        tool_calls = 0

        # Stage 1: Planning (single Claude call — no parallelism opportunity)
        if skip_planning:
            sub_questions = [question]
            if verbose:
                print("[planner] skipped — using question directly")
        else:
            if verbose:
                print("[planner] decomposing question...")
            sub_questions, reasoning = await asyncio.to_thread(
                self.planner.decompose, question, cost
            )
            tool_calls += 1
            if verbose:
                print(f"[planner] {len(sub_questions)} sub-questions | {reasoning}")
                for i, sq in enumerate(sub_questions, 1):
                    print(f"  {i}. {sq}")

        # Stage 2: Parallel search + semantic compression
        if verbose:
            print(f"[searcher] launching {len(sub_questions)} sub-question searches in parallel...")

        search_results = await asyncio.gather(
            *[self.searcher.search_async(sq, cost) for sq in sub_questions]
        )

        all_evidence: list[Evidence] = []
        for evidence, calls in search_results:
            tool_calls += calls
            all_evidence.extend(evidence)

        if verbose:
            good = sum(1 for e in all_evidence if e.search_successful)
            print(f"[searcher] {good} useful sources across all sub-questions")

        # Stage 3: Synthesis
        if verbose:
            print("[synthesizer] synthesizing answer...")
        answer_text, unanswered_sqs, sources = await asyncio.to_thread(
            self.synthesizer.synthesize, question, sub_questions, all_evidence, cost
        )
        tool_calls += 1

        # Stage 4: Verification
        claims = []
        unverified_texts: list[str] = []

        if not skip_verification:
            if verbose:
                print("[verifier] checking claims against evidence...")
            claims, unverified_texts = await asyncio.to_thread(
                self.verifier.verify, answer_text, all_evidence, sources, cost
            )
            tool_calls += 2
            if verbose:
                verified_count = sum(1 for c in claims if c.verified)
                print(
                    f"[verifier] {verified_count}/{len(claims)} verified, "
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
            cost=cost,
        )

    # ------------------------------------------------------------------
    # Sync wrapper (for scripts, eval harness, CLI)
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        skip_planning: bool = False,
        skip_verification: bool = False,
        verbose: bool = False,
    ) -> ResearchAnswer:
        """
        Sync entry point. Internally uses run_async() with asyncio.run().

        If you are already inside an async context (e.g. a Jupyter notebook),
        call run_async() directly with await instead.
        """
        return asyncio.run(
            self.run_async(question, skip_planning, skip_verification, verbose)
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
