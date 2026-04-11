"""Summarize eval run artifacts into a compact table.

Usage:
    python -m eval.summarize_results --latest
    python -m eval.summarize_results --file eval/results/20260411T120000Z.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re
import statistics

RESULTS_DIR = Path(__file__).parent / "results"
RUN_FILE_PATTERN = re.compile(r"^\d{8}T\d{6}Z\.json$")


def _latest_result_file() -> Path:
    files = sorted(
        p for p in RESULTS_DIR.glob("*.json") if RUN_FILE_PATTERN.match(p.name)
    )
    if not files:
        raise FileNotFoundError(
            "No timestamped result JSON files found in eval/results. "
            "Run `python -m eval.harness` first."
        )
    return files[-1]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _summarize(run: dict) -> list[dict]:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in run.get("results", []):
        if row.get("scores"):
            by_config[row["config"]].append(row)

    summary_rows: list[dict] = []
    for config, rows in by_config.items():
        citation = [r["scores"]["citation_accuracy"] for r in rows]
        completeness = [r["scores"]["completeness"] for r in rows]
        hallucination = [r["scores"]["hallucination_rate"] for r in rows]
        tool_calls = [r.get("tool_calls", 0) for r in rows]

        summary_rows.append(
            {
                "config": config,
                "citation_accuracy": round(statistics.mean(citation), 3),
                "completeness": round(statistics.mean(completeness), 3),
                "hallucination_rate": round(statistics.mean(hallucination), 3),
                "tool_calls": round(statistics.mean(tool_calls), 2),
                "n": len(rows),
            }
        )

    return sorted(summary_rows, key=lambda r: r["config"])


def _to_markdown(rows: list[dict]) -> str:
    header = "| Config | Citation Accuracy | Completeness | Hallucination Rate | Avg Tool Calls | N |"
    sep = "|---|---:|---:|---:|---:|---:|"
    body = []
    for r in rows:
        body.append(
            "| {config} | {citation_accuracy:.3f} | {completeness:.3f} | {hallucination_rate:.3f} | {tool_calls:.2f} | {n} |".format(
                **r
            )
        )
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument("--file", type=str, help="Path to a results JSON file")
    parser.add_argument("--latest", action="store_true", help="Use latest results JSON in eval/results")
    args = parser.parse_args()

    if args.file and args.latest:
        raise SystemExit("Use either --file or --latest, not both")

    if args.file:
        path = Path(args.file)
    else:
        path = _latest_result_file()

    run = _load(path)
    rows = _summarize(run)

    print(f"Run ID: {run.get('run_id', 'unknown')}")
    print(f"File: {path}")
    print(_to_markdown(rows))


if __name__ == "__main__":
    main()
