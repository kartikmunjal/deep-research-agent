"""
Evaluation Harness

Runs the research pipeline on every task in tasks.json across three
ablation configurations:

  1. no_plan_no_verify  - direct synthesis, no verification pass
  2. plan_no_verify     - planning + search, no verification
  3. plan_verify        - full pipeline (planning + search + verification)

Results are written to eval/results/{run_id}.json for offline analysis.

Usage:
    python -m eval.harness                             # full eval, all tasks (uses APIs)
    python -m eval.harness --offline                   # zero-cost synthetic run
    python -m eval.harness --category factual          # one category only
    python -m eval.harness --task-ids F01 M03          # specific tasks
    python -m eval.harness --configs plan_verify --dry-run  # print tasks only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.scoring import score_result


CONFIGS = {
    "no_plan_no_verify": {"skip_planning": True, "skip_verification": True},
    "plan_no_verify": {"skip_planning": False, "skip_verification": True},
    "plan_verify": {"skip_planning": False, "skip_verification": False},
}

RESULTS_DIR = Path(__file__).parent / "results"


def _get_git_commit() -> str:
    """Return current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _jitter(task_id: str, config_name: str, scale: float = 0.03) -> float:
    """Return deterministic jitter in [-scale, +scale] for stable synthetic fixtures."""
    digest = hashlib.sha256(f"{task_id}:{config_name}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    centered = (bucket / 999.0) * 2.0 - 1.0
    return centered * scale


def _offline_scores(task: dict, config_name: str) -> tuple[dict, int]:
    """Produce deterministic synthetic metrics for zero-cost smoke runs."""
    base = {
        "no_plan_no_verify": {
            "citation_accuracy": 0.60,
            "completeness": 0.54,
            "hallucination_rate": 0.34,
            "tool_calls": 3,
        },
        "plan_no_verify": {
            "citation_accuracy": 0.68,
            "completeness": 0.68,
            "hallucination_rate": 0.27,
            "tool_calls": 5,
        },
        "plan_verify": {
            "citation_accuracy": 0.73,
            "completeness": 0.69,
            "hallucination_rate": 0.10,
            "tool_calls": 6,
        },
    }[config_name].copy()

    jitter = _jitter(task["id"], config_name)
    base["citation_accuracy"] = _clamp(base["citation_accuracy"] + jitter)
    base["completeness"] = _clamp(base["completeness"] + jitter / 2)
    base["hallucination_rate"] = _clamp(base["hallucination_rate"] - jitter / 2)

    if task["category"] == "unanswerable":
        # Unanswerable tasks: safe behavior is abstention + low hallucination.
        base["completeness"] = _clamp(base["completeness"] - 0.15)
        base["hallucination_rate"] = _clamp(base["hallucination_rate"] - 0.05)
        if config_name == "plan_verify":
            # 10/11 ~= 91% uncertainty-signaling rate for full pipeline.
            base["uncertainty_reported"] = task["id"] not in {"U08", "U10"}
        elif config_name == "plan_no_verify":
            base["uncertainty_reported"] = task["id"] in {"U01", "U04", "U07", "U09"}
        else:
            base["uncertainty_reported"] = task["id"] in {"U01", "U04"}

    if task["category"] == "conflicting_evidence":
        # Conflicting evidence: primary signal is whether agent acknowledges the conflict
        # rather than collapsing to a single position.
        # Full pipeline should surface conflict most reliably; no-plan collapses more often.
        if config_name == "plan_verify":
            # Deterministic: 11/13 tasks acknowledge conflict (~85%)
            base["conflict_acknowledged"] = task["id"] not in {"C09", "C11"}
        elif config_name == "plan_no_verify":
            # ~62% acknowledge conflict (8/13)
            base["conflict_acknowledged"] = task["id"] in {
                "C01", "C03", "C05", "C06", "C08", "C10", "C12", "C13"
            }
        else:
            # ~38% without planning (5/13)
            base["conflict_acknowledged"] = task["id"] in {"C01", "C05", "C08", "C10", "C12"}

    if task["category"] == "gaia_l1":
        # GAIA L1 tasks are straightforward factual lookups.
        # Full pipeline should achieve near-perfect accuracy.
        if config_name == "plan_verify":
            # 11/12 correct (one miss from retrieval noise)
            base["gaia_accuracy"] = task["id"] != "G07"
            base["key_fact_recall"] = 0.92 + _jitter(task["id"], config_name, scale=0.05)
        elif config_name == "plan_no_verify":
            # 10/12 correct
            base["gaia_accuracy"] = task["id"] not in {"G07", "G11"}
            base["key_fact_recall"] = 0.85 + _jitter(task["id"], config_name, scale=0.05)
        else:
            # 8/12 correct without planning
            base["gaia_accuracy"] = task["id"] not in {"G03", "G05", "G07", "G11"}
            base["key_fact_recall"] = 0.72 + _jitter(task["id"], config_name, scale=0.05)
        base["key_fact_recall"] = _clamp(base.get("key_fact_recall", 0.8))

    # Synthetic cost estimate: proportional to tool_calls
    base["cost_usd"] = round(base["tool_calls"] * 0.018, 4)

    # Synthetic failure mode
    if base["hallucination_rate"] > 0.3:
        base["failure_mode"] = "genuine_hallucination"
    elif base["hallucination_rate"] > 0.0:
        base["failure_mode"] = "partial_hallucination"
    elif base["completeness"] < 0.5:
        base["failure_mode"] = "coverage_gap"
    else:
        base["failure_mode"] = "none"

    scores = {
        "citation_accuracy": round(base["citation_accuracy"], 3),
        "completeness": round(base["completeness"], 3),
        "hallucination_rate": round(base["hallucination_rate"], 3),
        "key_fact_recall": round(base.get("key_fact_recall", 0.8), 3),
        "tool_calls": base["tool_calls"],
        "cost_usd": base["cost_usd"],
        "failure_mode": base["failure_mode"],
    }
    if "uncertainty_reported" in base:
        scores["uncertainty_reported"] = base["uncertainty_reported"]
    if "conflict_acknowledged" in base:
        scores["conflict_acknowledged"] = base["conflict_acknowledged"]
    if "gaia_accuracy" in base:
        scores["gaia_accuracy"] = base["gaia_accuracy"]

    return scores, int(base["tool_calls"])


def load_tasks(
    category: str | None = None,
    task_ids: list[str] | None = None,
) -> list[dict]:
    tasks_path = Path(__file__).parent / "tasks.json"
    with open(tasks_path) as f:
        data = json.load(f)

    tasks = data["tasks"]

    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]

    return tasks


def run_eval(
    configs: list[str] | None = None,
    category: str | None = None,
    task_ids: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    model: str = "claude-sonnet-4-6",
    offline: bool = False,
) -> dict:
    configs = configs or list(CONFIGS.keys())
    tasks = load_tasks(category, task_ids)

    if not tasks:
        print("No tasks matched the filters.")
        return {}

    print(f"Running {len(tasks)} tasks x {len(configs)} configs = {len(tasks) * len(configs)} runs")

    if dry_run:
        for t in tasks:
            print(f"  [{t['id']}] {t['category']}: {t['question'][:70]}...")
        return {}

    pipeline = None
    if not offline:
        from src.agent.pipeline import ResearchPipeline

        pipeline = ResearchPipeline(model=model)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_results = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "model": model,
        "result_mode": "offline_fixture" if offline else "live_api",
        "configs": configs,
        "results": [],
    }

    if offline:
        print("[offline] Running deterministic synthetic eval. No external API calls will be made.")

    for task in tasks:
        for config_name in configs:
            config_kwargs = CONFIGS[config_name]
            print(f"\n[{config_name}] [{task['id']}] {task['question'][:60]}...")

            entry = {
                "task_id": task["id"],
                "category": task["category"],
                "question": task["question"],
                "config": config_name,
            }

            if offline:
                scores, tool_calls = _offline_scores(task, config_name)
                entry.update(
                    {
                        "elapsed_seconds": 0.0,
                        "tool_calls": tool_calls,
                        "num_sources": 0,
                        "num_sub_questions": 0,
                        "sub_questions": [],
                        "answer_text": "OFFLINE_SYNTHETIC_RESULT: no model call executed.",
                        "sources": [],
                        "num_claims": 0,
                        "unverified_claims": [],
                        "unanswered_sub_questions": [],
                        "scores": scores,
                        "scores_are_synthetic": True,
                        "error": None,
                    }
                )
                print(
                    f"  offline fixture | hallucination_rate={scores['hallucination_rate']:.2f} | "
                    f"completeness={scores['completeness']:.2f}"
                )
                all_results["results"].append(entry)
                continue

            try:
                t0 = time.time()
                answer = pipeline.run(
                    task["question"],
                    verbose=verbose,
                    **config_kwargs,
                )
                elapsed = time.time() - t0

                scores = score_result(answer, task)

                entry.update(
                    {
                        "elapsed_seconds": round(elapsed, 1),
                        "tool_calls": answer.tool_calls,
                        "num_sources": len(answer.sources),
                        "num_sub_questions": len(answer.sub_questions),
                        "sub_questions": answer.sub_questions,
                        "answer_text": answer.answer_text,
                        "sources": answer.sources,
                        "num_claims": len(answer.claims),
                        "unverified_claims": answer.unverified_claims,
                        "unanswered_sub_questions": answer.unanswered_sub_questions,
                        "scores": scores,
                        "scores_are_synthetic": False,
                        "error": None,
                    }
                )

                cost_str = (
                    f" | cost=${scores['cost_usd']:.4f}"
                    if "cost_usd" in scores else ""
                )
                print(
                    f"  ok {elapsed:.1f}s | sources={len(answer.sources)} | "
                    f"claims={len(answer.claims)} | "
                    f"hallucination_rate={scores['hallucination_rate']:.2f} | "
                    f"completeness={scores['completeness']:.2f}{cost_str}"
                )

            except Exception as e:
                entry.update(
                    {
                        "elapsed_seconds": None,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "scores": None,
                        "scores_are_synthetic": False,
                    }
                )
                print(f"  error: {e}")

            all_results["results"].append(entry)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults written to {out_path}")
    _print_summary(all_results)

    return all_results


def _print_summary(all_results: dict) -> None:
    from collections import defaultdict, Counter
    import statistics

    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in all_results["results"]:
        if r.get("scores"):
            by_config[r["config"]].append(r)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    header = (
        f"{'Config':<25} {'Cit.Acc':>8} {'Compl.':>8} {'Hall.%':>8} "
        f"{'ToolCalls':>10} {'Est.Cost':>10}"
    )
    print(header)
    print("-" * 80)

    for config_name in ["no_plan_no_verify", "plan_no_verify", "plan_verify"]:
        rows = by_config.get(config_name, [])
        if not rows:
            continue

        citation_accs = [r["scores"]["citation_accuracy"] for r in rows]
        completeness = [r["scores"]["completeness"] for r in rows]
        hall_rates = [r["scores"]["hallucination_rate"] for r in rows]
        tool_calls = [r["tool_calls"] for r in rows]
        costs = [r["scores"]["cost_usd"] for r in rows if "cost_usd" in r["scores"]]
        cost_str = f"${statistics.mean(costs):>8.4f}" if costs else f"{'n/a':>9}"

        print(
            f"{config_name:<25} "
            f"{statistics.mean(citation_accs):>8.2f} "
            f"{statistics.mean(completeness):>8.2f} "
            f"{statistics.mean(hall_rates)*100:>7.1f}% "
            f"{statistics.mean(tool_calls):>10.1f} "
            f"{cost_str:>10}"
        )

    print("=" * 80)

    # Behavioral summary rates — filter to plan_verify for meaningful comparisons
    pv_results = [r for r in all_results["results"]
                  if r.get("config") == "plan_verify" and r.get("scores")]

    unanswerable = [r for r in pv_results if r.get("category") == "unanswerable"]
    conflicting = [r for r in pv_results if r.get("category") == "conflicting_evidence"]
    gaia = [r for r in pv_results if r.get("category") == "gaia_l1"]

    if unanswerable:
        rate = sum(1 for r in unanswerable
                   if r["scores"].get("uncertainty_reported", False)) / len(unanswerable)
        print(f"\nUnanswerable uncertainty rate (plan_verify): {rate:.0%} ({len(unanswerable)} tasks)")

    if conflicting:
        rate = sum(1 for r in conflicting
                   if r["scores"].get("conflict_acknowledged", False)) / len(conflicting)
        print(f"Conflicting evidence acknowledged rate (plan_verify): {rate:.0%} ({len(conflicting)} tasks)")

    if gaia:
        rate = sum(1 for r in gaia if r["scores"].get("gaia_accuracy", False)) / len(gaia)
        print(f"GAIA L1 accuracy (plan_verify): {rate:.0%} ({len(gaia)} tasks)")

    # Failure mode taxonomy (plan_verify)
    if pv_results:
        failure_counts: Counter = Counter(
            r["scores"].get("failure_mode", "none") for r in pv_results
        )
        total = len(pv_results)
        print(f"\nFailure mode taxonomy — plan_verify ({total} tasks):")
        for mode in ["none", "partial_hallucination", "genuine_hallucination",
                     "coverage_gap", "retrieval_failure"]:
            count = failure_counts.get(mode, 0)
            if count > 0:
                print(f"  {mode:<30} {count:>3} ({count/total:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the deep research agent eval harness")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS.keys()))
    parser.add_argument(
        "--category",
        choices=["factual", "multi_hop", "unanswerable", "conflicting_evidence", "gaia_l1"],
    )
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--offline", action="store_true", help="Run deterministic synthetic eval; no API calls")
    args = parser.parse_args()

    run_eval(
        configs=args.configs,
        category=args.category,
        task_ids=args.task_ids,
        dry_run=args.dry_run,
        verbose=args.verbose,
        model=args.model,
        offline=args.offline,
    )
