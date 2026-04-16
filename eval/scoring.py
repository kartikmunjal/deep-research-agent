"""
Scoring Logic

Computes metrics per result by category:

  All categories:
    citation_accuracy   — fraction of claims the verifier traced to evidence
    completeness        — fraction of sub-questions addressed in the answer
    hallucination_rate  — fraction of claims the verifier could NOT trace to evidence
    key_fact_recall     — fraction of expected key_facts present in the answer text

  unanswerable tasks (additional):
    uncertainty_reported — whether the agent explicitly flagged it could not find
                           a definitive answer rather than fabricating one

  conflicting_evidence tasks (additional):
    conflict_acknowledged — whether the agent surfaced the evidential conflict
                            rather than collapsing to a single position

  gaia_l1 tasks (additional):
    gaia_accuracy — whether the answer contains the expected answer string (fuzzy match)

  Failure taxonomy (all tasks, plan_verify config):
    failure_mode — one of: none, retrieval_failure, coverage_gap,
                   partial_hallucination, genuine_hallucination

For unanswerable tasks, hallucination_rate + uncertainty_reported are primary.
For conflicting_evidence tasks, conflict_acknowledged is primary: an agent that
picks one side and presents it as settled consensus is failing the task even if
its claims are individually grounded.
For gaia_l1 tasks, gaia_accuracy is the primary signal; it measures whether the
agent can answer straightforward factual questions that any good retrieval agent
should handle correctly.
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
        citation_accuracy, completeness, hallucination_rate, tool_calls,
        key_fact_recall, failure_mode
        + uncertainty_reported  (unanswerable category only)
        + conflict_acknowledged (conflicting_evidence category only)
        + gaia_accuracy         (gaia_l1 category only)
    """
    category = task["category"]

    scores = {
        "citation_accuracy": round(answer.citation_accuracy, 3),
        "completeness": round(answer.completeness, 3),
        "hallucination_rate": round(answer.hallucination_rate, 3),
        "tool_calls": answer.tool_calls,
        "key_fact_recall": _key_fact_recall(answer, task),
        "failure_mode": categorize_failure(answer, task),
    }

    if answer.cost is not None:
        scores["cost_usd"] = round(answer.cost.estimate_usd, 4)

    if category == "unanswerable":
        scores["uncertainty_reported"] = _check_uncertainty_reported(answer)

    if category == "conflicting_evidence":
        scores["conflict_acknowledged"] = _check_conflict_acknowledged(answer)

    if category == "gaia_l1":
        scores["gaia_accuracy"] = _check_gaia_accuracy(answer, task)

    return scores


def _key_fact_recall(answer: ResearchAnswer, task: dict) -> float:
    """Fraction of expected key_facts present in the answer text (case-insensitive substring)."""
    key_facts = task.get("key_facts", [])
    if not key_facts:
        return 1.0
    text_lower = answer.answer_text.lower()
    found = sum(1 for kf in key_facts if kf.lower() in text_lower)
    return round(found / len(key_facts), 3)


def _check_gaia_accuracy(answer: ResearchAnswer, task: dict) -> bool:
    """True if the expected answer appears (case-insensitive) anywhere in the answer text."""
    expected = task.get("expected_answer", "")
    if not expected:
        return False
    return expected.lower() in answer.answer_text.lower()


def categorize_failure(answer: ResearchAnswer, task: dict) -> str:
    """
    Classify the primary failure mode for a result.

    Categories (mutually exclusive, most-severe-first priority):
      retrieval_failure   — no evidence was retrieved (all searches failed)
      coverage_gap        — some sub-questions had no evidence (partial retrieval)
      genuine_hallucination — hallucination_rate > 0.3 with reasonable completeness
      partial_hallucination — hallucination_rate > 0 but <= 0.3
      none                — no notable failure detected

    For unanswerable tasks, a low hallucination_rate is expected and is NOT a failure.
    """
    category = task.get("category", "")
    hall_rate = answer.hallucination_rate
    completeness = answer.completeness

    # Check for total retrieval failure
    all_failed = (
        len(answer.unverified_claims) == 0 and
        len(answer.claims) == 0 and
        completeness < 0.2
    )
    if all_failed:
        return "retrieval_failure"

    # Coverage gap: meaningful unanswered sub-questions
    if completeness < 0.6 and len(answer.unanswered_sub_questions) > 0:
        return "coverage_gap"

    # Skip hallucination categorisation for unanswerable tasks
    if category == "unanswerable":
        return "none"

    if hall_rate > 0.3:
        return "genuine_hallucination"

    if hall_rate > 0.0:
        return "partial_hallucination"

    return "none"


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
    behavioral summary rates for unanswerable and conflicting_evidence tasks,
    GAIA accuracy, and failure mode taxonomy distribution.
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

    _boolean_keys = {"uncertainty_reported", "conflict_acknowledged", "gaia_accuracy"}
    _skip_mean_keys = {"failure_mode"}

    def mean_scores(score_list: list[dict]) -> dict:
        if not score_list:
            return {}
        all_keys = set()
        for s in score_list:
            all_keys.update(s.keys())
        result = {}
        for k in all_keys:
            if k in _skip_mean_keys:
                continue
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
        "gaia_l1_accuracy": _gaia_accuracy_rate(results),
        "failure_taxonomy": _failure_taxonomy(results),
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


def _gaia_accuracy_rate(results: list[dict]) -> float:
    """Fraction of GAIA L1 tasks where the expected answer was found in the response."""
    rows = [
        r for r in results
        if r.get("category") == "gaia_l1" and r.get("scores")
    ]
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if r["scores"].get("gaia_accuracy", False))
    return round(correct / len(rows), 3)


def _failure_taxonomy(results: list[dict]) -> dict:
    """
    Count failure mode occurrences across all tasks (plan_verify config only,
    since other configs have structural gaps that aren't true failure modes).

    Returns dict mapping failure_mode -> count and rate.
    """
    from collections import Counter

    plan_verify_results = [
        r for r in results
        if r.get("config") == "plan_verify" and r.get("scores")
    ]
    if not plan_verify_results:
        # Fall back to all results if plan_verify not present
        plan_verify_results = [r for r in results if r.get("scores")]

    counts: Counter = Counter()
    for r in plan_verify_results:
        mode = r["scores"].get("failure_mode", "none")
        counts[mode] += 1

    total = len(plan_verify_results)
    taxonomy = {}
    for mode, count in sorted(counts.items()):
        taxonomy[mode] = {
            "count": count,
            "rate": round(count / total, 3) if total else 0.0,
        }
    return taxonomy
