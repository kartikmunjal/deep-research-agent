"""Parallel LangGraph implementation of the research agent orchestration layer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from tavily import TavilyClient, AsyncTavilyClient

from src.agent.models import Claim, Evidence, QueryCost, ResearchAnswer
from src.agent.planner import ResearchPlanner
from src.agent.searcher import ResearchSearcher
from src.agent.synthesizer import ResearchSynthesizer
from src.agent.verifier import ResearchVerifier
from .memory import build_memory_saver


class ResearchState(TypedDict, total=False):
    question: str
    active_question: str
    sub_questions: list[str]
    evidence: list[Evidence]
    draft_answer: str
    verified_answer: str
    coverage_gaps: list[str]
    search_failures: list[str]
    sources: list[dict]
    claims: list[Claim]
    unverified_claims: list[str]
    tool_calls: int
    replan_attempts: int
    skip_planning: bool
    skip_verification: bool
    plan_reasoning: str
    replan_reasoning: str


def route_after_search(state: ResearchState) -> Literal["replan_node", "synthesize_node"]:
    sub_questions = state.get("sub_questions", [])
    failures = state.get("search_failures", [])
    if not sub_questions:
        return "synthesize_node"
    if len(failures) > len(sub_questions) / 2 and state.get("replan_attempts", 0) < 1:
        return "replan_node"
    return "synthesize_node"


class LangGraphResearchPipeline:
    """StateGraph-based parallel implementation of the research pipeline."""

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

        self._anthropic = Anthropic(api_key=anthropic_key)
        self._async_anthropic = AsyncAnthropic(api_key=anthropic_key)
        self._tavily = TavilyClient(api_key=tavily_key)
        self._async_tavily = AsyncTavilyClient(api_key=tavily_key)
        self.model = model

        self.planner = ResearchPlanner(self._anthropic, model)
        self.searcher = ResearchSearcher(
            self._anthropic,
            self._tavily,
            model,
            results_per_query,
            async_anthropic_client=self._async_anthropic,
            async_tavily_client=self._async_tavily,
        )
        self.synthesizer = ResearchSynthesizer(self._anthropic, model)
        self.verifier = ResearchVerifier(self._anthropic, model)
        self.graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(ResearchState)
        builder.add_node("plan_node", self.plan_node)
        builder.add_node("search_node", self.search_node)
        builder.add_node("replan_node", self.replan_node)
        builder.add_node("synthesize_node", self.synthesize_node)
        builder.add_node("verify_node", self.verify_node)

        builder.add_edge(START, "plan_node")
        builder.add_edge("plan_node", "search_node")
        builder.add_conditional_edges("search_node", route_after_search)
        builder.add_edge("replan_node", "plan_node")
        builder.add_edge("synthesize_node", "verify_node")
        builder.add_edge("verify_node", END)

        return builder.compile(checkpointer=build_memory_saver())

    def plan_node(self, state: ResearchState) -> ResearchState:
        question = state.get("active_question", state["question"])
        if state.get("skip_planning"):
            return {
                "active_question": question,
                "sub_questions": [question],
                "plan_reasoning": "Planning skipped; using the original question directly.",
            }
        sub_questions, reasoning = self.planner.decompose(question)
        return {
            "active_question": question,
            "sub_questions": sub_questions,
            "plan_reasoning": reasoning,
            "tool_calls": state.get("tool_calls", 0) + 1,
        }

    def search_node(self, state: ResearchState) -> ResearchState:
        evidence, calls, failures = self.searcher.search_all(state.get("sub_questions", []))
        return {
            "evidence": evidence,
            "search_failures": failures,
            "tool_calls": state.get("tool_calls", 0) + calls,
        }

    def replan_node(self, state: ResearchState) -> ResearchState:
        reformulated, reasoning = self.planner.replan_question(
            state["question"],
            state.get("search_failures", []),
        )
        return {
            "active_question": reformulated,
            "sub_questions": [],
            "evidence": [],
            "search_failures": [],
            "replan_attempts": state.get("replan_attempts", 0) + 1,
            "replan_reasoning": reasoning,
            "tool_calls": state.get("tool_calls", 0) + 1,
        }

    def synthesize_node(self, state: ResearchState) -> ResearchState:
        answer_text, unanswered, sources = self.synthesizer.synthesize(
            state["question"],
            state.get("sub_questions", []),
            state.get("evidence", []),
        )
        return {
            "draft_answer": answer_text,
            "coverage_gaps": unanswered,
            "sources": sources,
            "tool_calls": state.get("tool_calls", 0) + 1,
        }

    def verify_node(self, state: ResearchState) -> ResearchState:
        if state.get("skip_verification"):
            return {
                "verified_answer": state.get("draft_answer", ""),
                "claims": [],
                "unverified_claims": [],
            }
        claims, unverified = self.verifier.verify(
            state.get("draft_answer", ""),
            state.get("evidence", []),
            state.get("sources", []),
        )
        return {
            "verified_answer": state.get("draft_answer", ""),
            "claims": claims,
            "unverified_claims": unverified,
            "tool_calls": state.get("tool_calls", 0) + 2,
        }

    def run(
        self,
        question: str,
        skip_planning: bool = False,
        skip_verification: bool = False,
        verbose: bool = False,
        thread_id: str = "research-session",
    ) -> ResearchAnswer:
        _ = verbose
        state = self.graph.invoke(
            {
                "question": question,
                "active_question": question,
                "sub_questions": [],
                "evidence": [],
                "draft_answer": "",
                "verified_answer": "",
                "coverage_gaps": [],
                "search_failures": [],
                "sources": [],
                "claims": [],
                "unverified_claims": [],
                "tool_calls": 0,
                "replan_attempts": 0,
                "skip_planning": skip_planning,
                "skip_verification": skip_verification,
            },
            {"configurable": {"thread_id": thread_id}},
        )
        return ResearchAnswer(
            question=question,
            sub_questions=state.get("sub_questions", []),
            evidence=state.get("evidence", []),
            answer_text=state.get("verified_answer") or state.get("draft_answer", ""),
            sources=state.get("sources", []),
            claims=state.get("claims", []),
            unverified_claims=state.get("unverified_claims", []),
            unanswered_sub_questions=state.get("coverage_gaps", []),
            tool_calls=state.get("tool_calls", 0),
            cost=QueryCost(),
        )

    def run_formatted(
        self,
        question: str,
        skip_planning: bool = False,
        skip_verification: bool = False,
        verbose: bool = False,
    ) -> str:
        answer = self.run(
            question,
            skip_planning=skip_planning,
            skip_verification=skip_verification,
            verbose=verbose,
        )
        return answer.answer_text
