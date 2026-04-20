"""Freeze a timestamped eval run into a replay benchmark artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.replay import freeze_run


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an eval run into a replay benchmark")
    parser.add_argument("--run-file", required=True, help="Source eval/results JSON file")
    parser.add_argument("--output", required=True, help="Output replay benchmark JSON path")
    parser.add_argument(
        "--benchmark-name",
        required=True,
        help="Stable name for the frozen benchmark, e.g. core_live_28",
    )
    parser.add_argument(
        "--allow-nonlive",
        action="store_true",
        help="Allow freezing non-live artifacts for schema testing only",
    )
    args = parser.parse_args()

    run_path = Path(args.run_file)
    run = _load(run_path)
    result_mode = run.get("result_mode")
    if result_mode != "live_api" and not args.allow_nonlive:
        raise SystemExit(
            "Refusing to freeze a non-live artifact. "
            "Use --allow-nonlive only for schema testing."
        )

    payload = freeze_run(run, benchmark_name=args.benchmark_name, source_file=str(run_path))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote replay benchmark: {out_path}")


if __name__ == "__main__":
    main()
