#!/usr/bin/env python3
"""Compare the original research pipeline to the LangGraph implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.harness import CONFIGS, _offline_scores, load_tasks
from eval.scoring import score_result


CORE_28_TASK_IDS = [
    *[f"F{i:02d}" for i in range(1, 11)],
    *[f"M{i:02d}" for i in range(1, 11)],
    *[f"U{i:02d}" for i in range(1, 9)],
]


def _aggregate(rows: list[dict]) -> dict[str, float]:
    return {
        "citation_accuracy": round(statistics.mean(r["citation_accuracy"] for r in rows), 3),
        "completeness": round(statistics.mean(r["completeness"] for r in rows), 3),
        "hallucination_rate": round(statistics.mean(r["hallucination_rate"] for r in rows), 3),
        "tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
        "n": len(rows),
    }


def _table(rows: list[dict]) -> str:
    headers = ["Architecture", "Config", "Citation", "Completeness", "Hallucination", "Tool Calls", "N"]
    body = []
    for row in rows:
        body.append([
            row["architecture"],
            row["config"],
            f"{row['citation_accuracy']:.3f}",
            f"{row['completeness']:.3f}",
            f"{row['hallucination_rate']:.3f}",
            f"{row['tool_calls']:.2f}",
            str(row["n"]),
        ])
    widths = [max(len(str(item[i])) for item in [headers] + body) for i in range(len(headers))]
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    return "\n".join([fmt.format(*headers), sep, *[fmt.format(*row) for row in body]])


def main() -> None:
    p = argparse.ArgumentParser(description="Compare original vs LangGraph deep-research architectures")
    p.add_argument("--offline", action="store_true", help="Use synthetic eval scores only")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--output", default="eval/results/architecture_comparison.json")
    args = p.parse_args()

    tasks = load_tasks(task_ids=CORE_28_TASK_IDS)
    if args.offline:
        pipelines = {"original": None, "langgraph": None}
    else:
        from src.agent.pipeline import ResearchPipeline
        from src.agent_langgraph.graph import LangGraphResearchPipeline

        pipelines = {
            "original": ResearchPipeline(model=args.model),
            "langgraph": LangGraphResearchPipeline(model=args.model),
        }

    per_arch_config: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for architecture, pipeline in pipelines.items():
        for task in tasks:
            for config_name, config_kwargs in CONFIGS.items():
                if args.offline:
                    scores, _ = _offline_scores(task, config_name)
                else:
                    answer = pipeline.run(task["question"], **config_kwargs, verbose=False)
                    scores = score_result(answer, task)
                per_arch_config[(architecture, config_name)].append(scores)

    summary_rows = []
    for (architecture, config_name), scores in sorted(per_arch_config.items()):
        metrics = _aggregate(scores)
        summary_rows.append(
            {
                "architecture": architecture,
                "config": config_name,
                **metrics,
            }
        )

    payload = {
        "task_ids": CORE_28_TASK_IDS,
        "offline": args.offline,
        "rows": summary_rows,
        "table": _table(summary_rows),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(payload["table"])


if __name__ == "__main__":
    main()
