"""Validate and summarize cross-model or temporal v3 replication artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def validate(artifacts: list[Path], kind: str) -> dict:
    rows = [json.loads(path.read_text()) | {"artifact": str(path)} for path in artifacts]
    if not rows:
        raise ValueError("At least one artifact is required")
    protocol = {row.get("protocol") for row in rows}
    fingerprints = {row.get("dataset_fingerprint") for row in rows}
    if protocol != {"safety-v3-adaptive-canary"} or len(fingerprints) != 1:
        raise ValueError("Artifacts must share the locked v3 protocol and dataset fingerprint")
    if any("completed_at_utc" not in row for row in rows):
        raise ValueError("Every artifact must be complete")
    if kind == "cross_model" and len({row["model"] for row in rows}) < 2:
        raise ValueError("Cross-model replication requires at least two distinct model IDs")
    if kind == "cross_provider" and len({row.get("provider", "anthropic") for row in rows}) < 2:
        raise ValueError("Cross-provider replication requires at least two distinct providers")
    if kind == "temporal":
        dates = {datetime.fromisoformat(row["completed_at_utc"]).date().isoformat() for row in rows}
        if len(dates) < 3:
            raise ValueError("Temporal replication requires at least three distinct UTC dates")
    summary = []
    for row in rows:
        metrics = row["configurations"]["hardened_v3"]["metrics"]
        summary.append({
            "artifact": row["artifact"], "model": row["model"],
            "provider": row.get("provider", "anthropic"),
            "completed_at_utc": row["completed_at_utc"],
            "adaptive_gbr": metrics["adaptive_gbr"],
            "fpr": metrics["fpr"],
            "provider_refusals": metrics["provider_refusals"],
        })
    return {
        "kind": kind, "protocol": next(iter(protocol)),
        "dataset_fingerprint": next(iter(fingerprints)), "n_trials": len(rows),
        "runs": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("cross_model", "cross_provider", "temporal"))
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.artifacts, args.kind)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
