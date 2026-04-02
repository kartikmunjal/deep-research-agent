#!/usr/bin/env python3
"""
Deep Research Agent — Demo

Run a research question through the full pipeline.

Usage:
    python demo.py "How does CRISPR-Cas9 work?"
    python demo.py --question "What caused the 2008 financial crisis?" --verbose
    python demo.py --question "..." --no-verify   # skip verification (faster)

Requires:
    ANTHROPIC_API_KEY and TAVILY_API_KEY set in environment or .env file
"""

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars may be set directly

from src.agent.pipeline import ResearchPipeline


DEFAULT_QUESTION = (
    "What is RLHF and how did it change the trajectory of large language model alignment?"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Deep Research Agent on a question",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Research question (uses a default if not provided)",
    )
    parser.add_argument("--question", "-q", dest="question_flag", help="Alternative way to pass the question")
    parser.add_argument("--no-plan", action="store_true", help="Skip planning step (ablation)")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification step (faster)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print pipeline progress")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model to use")
    args = parser.parse_args()

    question = args.question or args.question_flag or DEFAULT_QUESTION

    print(f"\nQuestion: {question}\n")
    print("=" * 70)

    pipeline = ResearchPipeline(model=args.model)

    result = pipeline.run(
        question,
        skip_planning=args.no_plan,
        skip_verification=args.no_verify,
        verbose=args.verbose or True,
    )

    print("\n" + "=" * 70)
    print(result.answer_text)

    if result.sources:
        print("\n## Sources")
        for s in result.sources:
            print(f"{s['number']}. {s['title']}")
            print(f"   {s['url']}")

    if result.unverified_claims:
        print(f"\n## Unverified Claims ({len(result.unverified_claims)})")
        for c in result.unverified_claims:
            print(f"  - {c}")

    if result.unanswered_sub_questions:
        print(f"\n## Coverage Gaps ({len(result.unanswered_sub_questions)})")
        for sq in result.unanswered_sub_questions:
            print(f"  - {sq}")

    print("\n" + "=" * 70)
    print(f"Stats: {result.tool_calls} API calls | "
          f"{len(result.sources)} sources | "
          f"{len(result.claims)} claims | "
          f"hallucination rate {result.hallucination_rate:.0%} | "
          f"completeness {result.completeness:.0%}")


if __name__ == "__main__":
    main()
