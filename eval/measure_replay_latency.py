"""Measure async search savings from frozen run artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _plan_verify_rows(run: dict) -> list[dict]:
    return [
        row for row in run.get("results", [])
        if row.get("config") == "plan_verify"
        and not row.get("error")
        and row.get("sub_question_timings")
        and row.get("stage_timings", {}).get("search_seconds") is not None
    ]


def summarize_latency(run: dict) -> dict:
    rows = _plan_verify_rows(run)
    if not rows:
        raise ValueError(
            "No plan_verify rows with timing data were found. "
            "Capture a live run with stage_timings/sub_question_timings first."
        )

    per_task = []
    for row in rows:
        async_search = float(row["stage_timings"]["search_seconds"])
        serial_search = sum(
            float(item.get("search_seconds", 0.0))
            for item in row.get("sub_question_timings", [])
        )
        speedup = (serial_search / async_search) if async_search > 0 else 0.0
        per_task.append(
            {
                "task_id": row["task_id"],
                "async_search_seconds": round(async_search, 4),
                "serial_search_estimate_seconds": round(serial_search, 4),
                "estimated_speedup": round(speedup, 3),
            }
        )

    return {
        "tasks": len(per_task),
        "mean_async_search_seconds": round(
            statistics.mean(item["async_search_seconds"] for item in per_task), 4
        ),
        "mean_serial_search_estimate_seconds": round(
            statistics.mean(item["serial_search_estimate_seconds"] for item in per_task), 4
        ),
        "mean_estimated_speedup": round(
            statistics.mean(item["estimated_speedup"] for item in per_task), 3
        ),
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure async-vs-serial search latency from frozen artifacts")
    parser.add_argument("--file", required=True, help="Path to a live_api or replay_benchmark JSON file")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    run = _load(Path(args.file))
    summary = summarize_latency(run)

    print(f"Tasks: {summary['tasks']}")
    print(f"Mean async search seconds: {summary['mean_async_search_seconds']:.4f}")
    print(
        "Mean serial search estimate seconds: "
        f"{summary['mean_serial_search_estimate_seconds']:.4f}"
    )
    print(f"Mean estimated speedup: {summary['mean_estimated_speedup']:.3f}x")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote results: {out_path}")


if __name__ == "__main__":
    main()
