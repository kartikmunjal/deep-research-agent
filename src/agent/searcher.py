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
"""

import re
from anthropic import Anthropic
from tavily import TavilyClient

from .models import Evidence


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
    def __init__(
        self,
        anthropic_client: Anthropic,
        tavily_client: TavilyClient,
        model: str,
        results_per_query: int = 4,
        max_content_chars: int = 3000,
    ):
        self.anthropic = anthropic_client
        self.tavily = tavily_client
        self.model = model
        self.results_per_query = results_per_query
        self.max_content_chars = max_content_chars

    def search(self, sub_question: str) -> tuple[list[Evidence], int]:
        """
        Search for evidence relevant to a sub-question.

        Returns:
            (evidence_list, tool_call_count) tuple
        """
        tool_calls = 0
        evidence, calls = self._search_and_extract(sub_question, sub_question)
        tool_calls += calls

        if not evidence:
            # Reformulate and retry once
            reformulated = self._reformulate(sub_question)
            tool_calls += 1
            evidence, calls = self._search_and_extract(sub_question, reformulated)
            tool_calls += calls

        if not evidence:
            # Mark as unresolved — pipeline will surface this in the final answer
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

    def _search_and_extract(
        self, sub_question: str, query: str
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
        except Exception:
            return [], tool_calls

        evidence = []
        for r in results.get("results", []):
            content = (r.get("raw_content") or r.get("content") or "").strip()
            if not content:
                continue

            content = content[: self.max_content_chars]
            extracted = self._extract_relevant(
                sub_question=sub_question,
                title=r.get("title", ""),
                content=content,
            )
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

    def _extract_relevant(self, sub_question: str, title: str, content: str) -> str:
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(
                        sub_question=sub_question,
                        title=title,
                        content=content,
                    ),
                }
            ],
        )
        return response.content[0].text.strip()

    def _reformulate(self, query: str) -> str:
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": REFORMULATE_PROMPT.format(query=query),
                }
            ],
        )
        return response.content[0].text.strip()
