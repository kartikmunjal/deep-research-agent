"""
Scoring Logic

Computes four metrics per result:

  citation_accuracy   — fraction of claims the verifier traced to evidence
  completeness        — fraction of sub-questions addressed in the answer
  hallucination_rate  — fraction of claims the verifier could NOT trace to evidence
  uncertainty_reported — (unanswerable tasks only) whether the agent explicitly
                         flagged it could not find a definitive answer

For unanswerable tasks, hallucination_rate is the primary metric: a good agent
should surface uncertainty rather than fabricate. A zero hallucination_rate on
an unanswerable task means the agent correctly refused to make unsupported claims.
"""

import re
from src.agent.models import ResearchAnswer


# Keywords that signal appropriate uncertainty reporting
UNCERTAINTY_PHRASES = [
    "could not be found",
    "could not find",
    "not publicly available",
    "not publicly disclosed",
    "no public information",
    "unable to verify",
    "cannot verify",
    "no reliable sources",
    "not accessible",
    "undisclosed",
    "classified",
    "not disclosed",
    "proprietary",
    "not found in available sources",
    "coverage gap",
]


def score_result(answer: ResearchAnswer, task: dict) -> dict:
    """
    Compute all metrics for a single research result.

    Returns a dict with keys:
        citation_accuracy, completeness, hallucination_rate,
        uncertainty_reported (for unanswerable tasks), tool_calls
    """
    category = task["category"]

    citation_accuracy = answer.citation_accuracy
    completeness = answer.completeness
    hallucination_rate = answer.hallucination_rate

    scores = {
        "citation_accuracy": round(citation_accuracy, 3),
        "completeness": round(completeness, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "tool_calls": answer.tool_calls,
    }

    if category == "unanswerable":
        scores["uncertainty_reported"] = _check_uncertainty_reported(answer)

    return scores


def _check_uncertainty_reported(answer: ResearchAnswer) -> bool:
    """
    Check whether the agent explicitly flagged uncertainty for an unanswerable question.

    Returns True if the answer contains explicit uncertainty language OR if the
    agent listed unanswered sub-questions, indicating it recognized the gap.
    """
    text_lower = answer.answer_text.lower()
    has_phrase = any(phrase in text_lower for phrase in UNCERTAINTY_PHRASES)
    has_unanswered = len(answer.unanswered_sub_questions) > 0
    return has_phrase or has_unanswered


def aggregate_results(results: list[dict]) -> dict:
    """
    Aggregate scores across multiple results for reporting.

    Args:
        results: list of result dicts from the harness (each has a 'scores' key)

    Returns:
        dict with mean metrics per config and per category
    """
    from collections import defaultdict
    import statistics

    by_config: dict[str, list] = defaultdict(list)
    by_category: dict[str, list] = defaultdict(list)

    for r in results:
        if not r.get("scores"):
            continue
        by_config[r["config"]].append(r["scores"])
        by_category[r["category"]].append(r["scores"])

    def mean_scores(score_list: list[dict]) -> dict:
        if not score_list:
            return {}
        keys = [k for k in score_list[0] if k != "uncertainty_reported"]
        result = {}
        for k in keys:
            vals = [s[k] for s in score_list if k in s]
            result[f"mean_{k}"] = round(statistics.mean(vals), 3)
        return result

    return {
        "by_config": {k: mean_scores(v) for k, v in by_config.items()},
        "by_category": {k: mean_scores(v) for k, v in by_category.items()},
        "unanswerable_uncertainty_rate": _unanswerable_uncertainty_rate(results),
    }


def _unanswerable_uncertainty_rate(results: list[dict]) -> float:
    unanswerable = [
        r for r in results
        if r.get("category") == "unanswerable" and r.get("scores")
    ]
    if not unanswerable:
        return 0.0
    flagged = sum(
        1 for r in unanswerable if r["scores"].get("uncertainty_reported", False)
    )
    return round(flagged / len(unanswerable), 3)
