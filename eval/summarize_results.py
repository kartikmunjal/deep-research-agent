"""Summarize eval run artifacts into compact tables.

Usage:
    python -m eval.summarize_results --latest
    python -m eval.summarize_results --latest --mode live_api --require-live
    python -m eval.summarize_results --file eval/results/20260411T120000Z.json
    python -m eval.summarize_results --latest --export-json eval/results/summary.json
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


def _run_mode(run: dict) -> str:
    return str(run.get("result_mode", "unknown"))


def _latest_result_file(mode: str = "any") -> Path:
    candidates = sorted(
        p for p in RESULTS_DIR.glob("*.json") if RUN_FILE_PATTERN.match(p.name)
    )
    if mode == "any":
        if not candidates:
            raise FileNotFoundError(
                "No timestamped result JSON files found in eval/results. "
                "Run `python -m eval.harness` first."
            )
        return candidates[-1]

    filtered: list[Path] = []
    for path in candidates:
        run = _load(path)
        if _run_mode(run) == mode:
            filtered.append(path)

    if not filtered:
        raise FileNotFoundError(
            f"No timestamped result JSON files found in eval/results with result_mode={mode}."
        )

    return filtered[-1]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _metric_means(rows: list[dict]) -> dict[str, float]:
    citation = [r["scores"]["citation_accuracy"] for r in rows]
    completeness = [r["scores"]["completeness"] for r in rows]
    hallucination = [r["scores"]["hallucination_rate"] for r in rows]
    tool_calls = [r.get("tool_calls", 0) for r in rows]
    return {
        "citation_accuracy": round(statistics.mean(citation), 3),
        "completeness": round(statistics.mean(completeness), 3),
        "hallucination_rate": round(statistics.mean(hallucination), 3),
        "tool_calls": round(statistics.mean(tool_calls), 2),
        "n": len(rows),
    }


def _summarize(run: dict) -> list[dict]:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in run.get("results", []):
        if row.get("scores"):
            by_config[row["config"]].append(row)

    summary_rows = []
    for config, rows in by_config.items():
        metrics = _metric_means(rows)
        summary_rows.append({"config": config, **metrics})

    return sorted(summary_rows, key=lambda r: r["config"])


def _summarize_by_category(run: dict) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in run.get("results", []):
        if row.get("scores"):
            by_category[row["category"]].append(row)

    summary_rows = []
    for category, rows in by_category.items():
        metrics = _metric_means(rows)
        summary_rows.append({"category": category, **metrics})

    return sorted(summary_rows, key=lambda r: r["category"])


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


def _to_markdown_category(rows: list[dict]) -> str:
    header = "| Category | Citation Accuracy | Completeness | Hallucination Rate | Avg Tool Calls | N |"
    sep = "|---|---:|---:|---:|---:|---:|"
    body = []
    for r in rows:
        body.append(
            "| {category} | {citation_accuracy:.3f} | {completeness:.3f} | {hallucination_rate:.3f} | {tool_calls:.2f} | {n} |".format(
                **r
            )
        )
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument("--file", type=str, help="Path to a results JSON file")
    parser.add_argument("--latest", action="store_true", help="Use latest results JSON in eval/results")
    parser.add_argument(
        "--mode",
        choices=["any", "live_api", "offline_fixture"],
        default="any",
        help="Filter latest run selection by result_mode when using --latest",
    )
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Exit non-zero if the selected run is not result_mode=live_api",
    )
    parser.add_argument("--export-json", type=str, help="Optional path to write machine-readable summary")
    args = parser.parse_args()

    if args.file and args.latest:
        raise SystemExit("Use either --file or --latest, not both")

    try:
        if args.file:
            path = Path(args.file)
        else:
            path = _latest_result_file(mode=args.mode)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    run = _load(path)
    mode = _run_mode(run)

    if args.require_live and mode != "live_api":
        raise SystemExit(
            f"Selected run is result_mode={mode}. Re-run with --mode live_api after a paid benchmark run."
        )

    by_config = _summarize(run)
    by_category = _summarize_by_category(run)

    print(f"Run ID: {run.get('run_id', 'unknown')}")
    print(f"File: {path}")
    print(f"Result Mode: {mode}")
    if mode == "offline_fixture":
        print("WARNING: offline_fixture metrics are synthetic smoke-run outputs, not benchmark claims.")

    print("\nBy Config")
    print(_to_markdown(by_config))
    print("\nBy Category")
    print(_to_markdown_category(by_category))

    if args.export_json:
        payload = {
            "run_id": run.get("run_id", "unknown"),
            "source_file": str(path),
            "result_mode": mode,
            "by_config": by_config,
            "by_category": by_category,
        }
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote summary JSON: {out_path}")


if __name__ == "__main__":
    main()
