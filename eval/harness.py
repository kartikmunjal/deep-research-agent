"""
Evaluation Harness

Runs the research pipeline on every task in tasks.json across three
ablation configurations:

  1. no_plan_no_verify  - direct synthesis, no verification pass
  2. plan_no_verify     - planning + search, no verification
  3. plan_verify        - full pipeline (planning + search + verification)

Results are written to eval/results/{run_id}.json for offline analysis.

Usage:
    python -m eval.harness                    # full eval, all tasks
    python -m eval.harness --category factual # one category only
    python -m eval.harness --task-ids F01 M03 # specific tasks
    python -m eval.harness --configs plan_verify --dry-run  # print tasks only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.pipeline import ResearchPipeline
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

    pipeline = ResearchPipeline(model=model)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_results = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "model": model,
        "configs": configs,
        "results": [],
    }

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
                        "error": None,
                    }
                )

                print(
                    f"  ok {elapsed:.1f}s | sources={len(answer.sources)} | "
                    f"claims={len(answer.claims)} | "
                    f"hallucination_rate={scores['hallucination_rate']:.2f} | "
                    f"completeness={scores['completeness']:.2f}"
                )

            except Exception as e:
                entry.update(
                    {
                        "elapsed_seconds": None,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "scores": None,
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
    from collections import defaultdict
    import statistics

    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in all_results["results"]:
        if r.get("scores"):
            by_config[r["config"]].append(r)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    header = f"{'Config':<25} {'Cit.Acc':>8} {'Compl.':>8} {'Hall.%':>8} {'ToolCalls':>10}"
    print(header)
    print("-" * 70)

    for config_name in ["no_plan_no_verify", "plan_no_verify", "plan_verify"]:
        rows = by_config.get(config_name, [])
        if not rows:
            continue

        citation_accs = [r["scores"]["citation_accuracy"] for r in rows]
        completeness = [r["scores"]["completeness"] for r in rows]
        hall_rates = [r["scores"]["hallucination_rate"] for r in rows]
        tool_calls = [r["tool_calls"] for r in rows]

        print(
            f"{config_name:<25} "
            f"{statistics.mean(citation_accs):>8.2f} "
            f"{statistics.mean(completeness):>8.2f} "
            f"{statistics.mean(hall_rates)*100:>7.1f}% "
            f"{statistics.mean(tool_calls):>10.1f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the deep research agent eval harness")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS.keys()))
    parser.add_argument("--category", choices=["factual", "multi_hop", "unanswerable"])
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    run_eval(
        configs=args.configs,
        category=args.category,
        task_ids=args.task_ids,
        dry_run=args.dry_run,
        verbose=args.verbose,
        model=args.model,
    )
