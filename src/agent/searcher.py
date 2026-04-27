"""
Research Searcher

Two-stage retrieval for each sub-question:
  1. Tavily API fetches top N web results
  2. Claude performs semantic compression — extracting only the sentences
     from each result that directly address the sub-question

The compression step is the key context management mechanism. A naive
implementation that passes full article text to the synthesizer would
either overflow the context window or dilute the signal with irrelevant
prose. Instead, we extract 2-5 relevant sentences per source and store
those extracts alongside their URLs. This keeps the evidence set tight
regardless of how many sources we retrieve.

Failure recovery: if a Tavily search returns no useful results, the
searcher reformulates the query (strips filler words, tries synonyms)
and retries once. If the retry also fails, the sub-question is marked
as unresolved and explicitly flagged in the final answer.

Async interface: search_async() parallelises sub-question retrieval via
asyncio.gather in the pipeline. The sync search() method remains for
single-question use and testing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from tavily import TavilyClient, AsyncTavilyClient

from .models import Evidence, QueryCost


EXTRACT_PROMPT = """You are extracting relevant evidence from a web article.

Sub-question being researched: {sub_question}

Article title: {title}
Article content:
{content}

Extract 2-5 sentences from the article that most directly answer or inform the sub-question.
Copy them verbatim — do not paraphrase.
If the article contains NO relevant information, respond with exactly: NO_RELEVANT_CONTENT

Respond with only the extracted sentences (or NO_RELEVANT_CONTENT). No preamble."""


REFORMULATE_PROMPT = """The following search query returned no useful results. Reformulate it to be more specific and searchable.

Original query: {query}

Respond with only the reformulated query. No explanation."""


class ResearchSearcher:
    """Retrieves candidate pages and extracts semantically relevant evidence."""

    def __init__(
        self,
        anthropic_client: Anthropic,
        tavily_client: TavilyClient,
        model: str,
        results_per_query: int = 4,
        max_content_chars: int = 3000,
        async_anthropic_client: AsyncAnthropic | None = None,
        async_tavily_client: AsyncTavilyClient | None = None,
    ):
        self.anthropic = anthropic_client
        self.tavily = tavily_client
        self.model = model
        self.results_per_query = results_per_query
        self.max_content_chars = max_content_chars
        self._async_anthropic = async_anthropic_client
        self._async_tavily = async_tavily_client

    # ------------------------------------------------------------------
    # Sync interface (single sub-question, blocking)
    # ------------------------------------------------------------------

    def search(self, sub_question: str, cost: QueryCost | None = None) -> tuple[list[Evidence], int]:
        """Search and extract evidence for one sub-question (blocking)."""
        tool_calls = 0
        evidence, calls = self._search_and_extract(sub_question, sub_question, cost)
        tool_calls += calls

        if not evidence:
            reformulated = self._reformulate(sub_question, cost)
            tool_calls += 1
            evidence, calls = self._search_and_extract(sub_question, reformulated, cost)
            tool_calls += calls

        if not evidence:
            evidence = [
                Evidence(
                    url="",
                    title="",
                    extracted_text="",
                    sub_question=sub_question,
                    search_successful=False,
                )
            ]

        return evidence, tool_calls

    def search_all(
        self,
        sub_questions: list[str],
        cost: QueryCost | None = None,
    ) -> tuple[list[Evidence], int, list[str]]:
        """Search all sub-questions and return evidence, tool calls, and failures."""
        return asyncio.run(self.search_all_async(sub_questions, cost))

    def _search_and_extract(
        self, sub_question: str, query: str, cost: QueryCost | None
    ) -> tuple[list[Evidence], int]:
        tool_calls = 0
        try:
            results = self.tavily.search(
                query=query,
                max_results=self.results_per_query,
                search_depth="advanced",
                include_raw_content=True,
            )
            tool_calls += 1
            if cost is not None:
                cost.add_tavily()
        except Exception:
            return [], tool_calls

        evidence = []
        for r in results.get("results", []):
            content = (r.get("raw_content") or r.get("content") or "").strip()
            if not content:
                continue
            content = content[: self.max_content_chars]
            extracted = self._extract_relevant(sub_question, r.get("title", ""), content, cost)
            tool_calls += 1
            if extracted and extracted != "NO_RELEVANT_CONTENT":
                evidence.append(
                    Evidence(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        extracted_text=extracted,
                        sub_question=sub_question,
                        search_successful=True,
                    )
                )

        return evidence, tool_calls

    def _extract_relevant(
        self, sub_question: str, title: str, content: str, cost: QueryCost | None
    ) -> str:
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(
                        sub_question=sub_question, title=title, content=content
                    ),
                }
            ],
        )
        if cost is not None:
            cost.add_response(response.usage)
        return response.content[0].text.strip()

    def _reformulate(self, query: str, cost: QueryCost | None) -> str:
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=128,
            messages=[{"role": "user", "content": REFORMULATE_PROMPT.format(query=query)}],
        )
        if cost is not None:
            cost.add_response(response.usage)
        return response.content[0].text.strip()

    # ------------------------------------------------------------------
    # Async interface (single sub-question, non-blocking)
    # Called via asyncio.gather in the pipeline for parallel retrieval.
    # ------------------------------------------------------------------

    async def search_async(
        self, sub_question: str, cost: QueryCost | None = None
    ) -> tuple[list[Evidence], int]:
        """Non-blocking search for one sub-question. Use asyncio.gather for parallelism."""
        tool_calls = 0
        evidence, calls = await self._search_and_extract_async(sub_question, sub_question, cost)
        tool_calls += calls

        if not evidence:
            reformulated = await self._reformulate_async(sub_question, cost)
            tool_calls += 1
            evidence, calls = await self._search_and_extract_async(sub_question, reformulated, cost)
            tool_calls += calls

        if not evidence:
            evidence = [
                Evidence(
                    url="",
                    title="",
                    extracted_text="",
                    sub_question=sub_question,
                    search_successful=False,
                )
            ]

        return evidence, tool_calls

    async def search_all_async(
        self,
        sub_questions: list[str],
        cost: QueryCost | None = None,
    ) -> tuple[list[Evidence], int, list[str]]:
        """Parallel search across all sub-questions."""
        results = await asyncio.gather(*[self.search_async(sq, cost) for sq in sub_questions])
        all_evidence: list[Evidence] = []
        total_calls = 0
        failures: list[str] = []
        for sub_question, (evidence, calls) in zip(sub_questions, results):
            total_calls += calls
            all_evidence.extend(evidence)
            if not any(item.search_successful for item in evidence):
                failures.append(sub_question)
        return all_evidence, total_calls, failures

    async def _search_and_extract_async(
        self, sub_question: str, query: str, cost: QueryCost | None
    ) -> tuple[list[Evidence], int]:
        tool_calls = 0
        try:
            if self._async_tavily is not None:
                results = await self._async_tavily.search(
                    query=query,
                    max_results=self.results_per_query,
                    search_depth="advanced",
                    include_raw_content=True,
                )
            else:
                # Fall back to running sync client in a thread
                results = await asyncio.to_thread(
                    self.tavily.search,
                    query=query,
                    max_results=self.results_per_query,
                    search_depth="advanced",
                    include_raw_content=True,
                )
            tool_calls += 1
            if cost is not None:
                cost.add_tavily()
        except Exception:
            return [], tool_calls

        tasks = []
        raw_results = results.get("results", [])
        for r in raw_results:
            content = (r.get("raw_content") or r.get("content") or "").strip()
            if not content:
                continue
            content = content[: self.max_content_chars]
            tasks.append(
                self._extract_relevant_async(sub_question, r.get("title", ""), content, cost, r)
            )

        extracted_results = await asyncio.gather(*tasks, return_exceptions=True)

        evidence = []
        for item in extracted_results:
            if isinstance(item, Exception):
                continue
            tool_calls += 1
            ev, r = item
            if ev and ev != "NO_RELEVANT_CONTENT":
                evidence.append(
                    Evidence(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        extracted_text=ev,
                        sub_question=sub_question,
                        search_successful=True,
                    )
                )

        return evidence, tool_calls

    async def _extract_relevant_async(
        self,
        sub_question: str,
        title: str,
        content: str,
        cost: QueryCost | None,
        raw_result: dict,
    ) -> tuple[str, dict]:
        prompt = EXTRACT_PROMPT.format(sub_question=sub_question, title=title, content=content)
        if self._async_anthropic is not None:
            response = await self._async_anthropic.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            response = await asyncio.to_thread(
                self.anthropic.messages.create,
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
        if cost is not None:
            cost.add_response(response.usage)
        return response.content[0].text.strip(), raw_result

    async def _reformulate_async(self, query: str, cost: QueryCost | None) -> str:
        prompt = REFORMULATE_PROMPT.format(query=query)
        if self._async_anthropic is not None:
            response = await self._async_anthropic.messages.create(
                model=self.model,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            response = await asyncio.to_thread(
                self.anthropic.messages.create,
                model=self.model,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
        if cost is not None:
            cost.add_response(response.usage)
        return response.content[0].text.strip()
