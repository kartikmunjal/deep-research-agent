"""
Scoring Logic

Computes metrics per result by category:

  All categories:
    citation_accuracy   — fraction of claims the verifier traced to evidence
    completeness        — fraction of sub-questions addressed in the answer
    hallucination_rate  — fraction of claims the verifier could NOT trace to evidence

  unanswerable tasks (additional):
    uncertainty_reported — whether the agent explicitly flagged it could not find
                           a definitive answer rather than fabricating one

  conflicting_evidence tasks (additional):
    conflict_acknowledged — whether the agent surfaced the evidential conflict
                            rather than collapsing to a single position

For unanswerable tasks, hallucination_rate + uncertainty_reported are primary.
For conflicting_evidence tasks, conflict_acknowledged is primary: an agent that
picks one side and presents it as settled consensus is failing the task even if
its claims are individually grounded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.models import ResearchAnswer


# Phrases indicating appropriate uncertainty for unanswerable questions
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

# Phrases indicating the agent acknowledged a genuine conflict between sources.
# Two or more of these in close proximity signals adjudication behavior.
CONFLICT_PHRASES = [
    "however",
    "on the other hand",
    "in contrast",
    "conversely",
    "while others",
    "while some",
    "some studies",
    "other studies",
    "some researchers",
    "other researchers",
    "conflicting",
    "contradictory",
    "contested",
    "disputed",
    "mixed evidence",
    "evidence is mixed",
    "debate",
    "disagree",
    "not settled",
    "both positions",
    "two views",
    "different conclusions",
    "researchers disagree",
    "evidence suggests",
    "not conclusive",
    "no consensus",
]

# Phrases that signal false-consensus collapse — a bad outcome on conflicting tasks.
# If only these appear with no conflict phrases, the agent likely picked a side.
FALSE_CONSENSUS_PHRASES = [
    "the evidence clearly shows",
    "it is well established",
    "research conclusively",
    "the consensus is clear",
    "definitively shows",
    "it is certain",
]


def score_result(answer: ResearchAnswer, task: dict) -> dict:
    """
    Compute all metrics for a single research result.

    Returns a dict with keys:
        citation_accuracy, completeness, hallucination_rate, tool_calls
        + uncertainty_reported  (unanswerable category only)
        + conflict_acknowledged (conflicting_evidence category only)
    """
    category = task["category"]

    scores = {
        "citation_accuracy": round(answer.citation_accuracy, 3),
        "completeness": round(answer.completeness, 3),
        "hallucination_rate": round(answer.hallucination_rate, 3),
        "tool_calls": answer.tool_calls,
    }

    if category == "unanswerable":
        scores["uncertainty_reported"] = _check_uncertainty_reported(answer)

    if category == "conflicting_evidence":
        scores["conflict_acknowledged"] = _check_conflict_acknowledged(answer)

    return scores


def _check_uncertainty_reported(answer: ResearchAnswer) -> bool:
    """True if the answer contains explicit uncertainty language or lists unanswered sub-questions."""
    text_lower = answer.answer_text.lower()
    has_phrase = any(phrase in text_lower for phrase in UNCERTAINTY_PHRASES)
    has_unanswered = len(answer.unanswered_sub_questions) > 0
    return has_phrase or has_unanswered


def _check_conflict_acknowledged(answer: ResearchAnswer) -> bool:
    """
    True if the answer surfaces a genuine evidential conflict rather than collapsing
    to a single position.

    Heuristic: the answer must contain at least 2 distinct conflict-signaling phrases,
    OR contain one strong conflict word ("contested", "disputed", "no consensus",
    "researchers disagree"). A single "however" is insufficient — that can appear in
    a one-sided answer. False-consensus language without any conflict phrases is a
    strong negative signal.
    """
    text_lower = answer.answer_text.lower()

    strong_conflict = {
        "contested", "disputed", "no consensus", "researchers disagree",
        "debate", "conflicting", "contradictory", "mixed evidence",
        "evidence is mixed", "not settled",
    }

    # Count how many distinct conflict phrases appear
    found_phrases = [p for p in CONFLICT_PHRASES if p in text_lower]
    found_strong = [p for p in strong_conflict if p in text_lower]

    has_false_consensus = any(p in text_lower for p in FALSE_CONSENSUS_PHRASES)

    if has_false_consensus and not found_strong:
        return False

    # At least one strong conflict phrase, or at least 2 general conflict phrases
    return len(found_strong) >= 1 or len(found_phrases) >= 2


def aggregate_results(results: list[dict]) -> dict:
    """
    Aggregate scores across multiple results for reporting.

    Returns dict with mean metrics per config and per category, plus
    behavioral summary rates for unanswerable and conflicting_evidence tasks.
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

    _boolean_keys = {"uncertainty_reported", "conflict_acknowledged"}

    def mean_scores(score_list: list[dict]) -> dict:
        if not score_list:
            return {}
        all_keys = set()
        for s in score_list:
            all_keys.update(s.keys())
        result = {}
        for k in all_keys:
            vals = [s[k] for s in score_list if k in s]
            if not vals:
                continue
            if k in _boolean_keys:
                result[f"rate_{k}"] = round(sum(vals) / len(vals), 3)
            else:
                result[f"mean_{k}"] = round(statistics.mean(vals), 3)
        return result

    return {
        "by_config": {k: mean_scores(v) for k, v in by_config.items()},
        "by_category": {k: mean_scores(v) for k, v in by_category.items()},
        "unanswerable_uncertainty_rate": _unanswerable_uncertainty_rate(results),
        "conflicting_evidence_acknowledged_rate": _conflict_acknowledged_rate(results),
    }


def _unanswerable_uncertainty_rate(results: list[dict]) -> float:
    rows = [
        r for r in results
        if r.get("category") == "unanswerable" and r.get("scores")
    ]
    if not rows:
        return 0.0
    flagged = sum(1 for r in rows if r["scores"].get("uncertainty_reported", False))
    return round(flagged / len(rows), 3)


def _conflict_acknowledged_rate(results: list[dict]) -> float:
    rows = [
        r for r in results
        if r.get("category") == "conflicting_evidence" and r.get("scores")
    ]
    if not rows:
        return 0.0
    acknowledged = sum(1 for r in rows if r["scores"].get("conflict_acknowledged", False))
    return round(acknowledged / len(rows), 3)
