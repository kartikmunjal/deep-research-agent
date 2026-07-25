"""Blinded two-reviewer annotation export and agreement analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval" / "safety_v3" / "dataset.json"


def export(path: Path) -> None:
    data = json.loads(DATASET.read_text())
    benign = [x for x in data["items"] if x["split"] == "benign"]
    rows = sorted(benign, key=lambda x: hashlib.sha256(x["id"].encode()).hexdigest())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "text", "label", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"blind_id": hashlib.sha256(row["id"].encode()).hexdigest()[:12],
                             "text": row["text"], "label": "", "reason": ""})


def analyze(first: Path, second: Path, output: Path) -> None:
    def read(path: Path) -> dict[str, dict]:
        return {r["blind_id"]: r for r in csv.DictReader(path.open())}
    a, b = read(first), read(second)
    if set(a) != set(b):
        raise SystemExit("Reviewer files contain different blind IDs")
    allowed = {"benign", "unsafe", "ambiguous"}
    pairs = []
    for key in sorted(a):
        if a[key]["label"] not in allowed or b[key]["label"] not in allowed:
            raise SystemExit(f"Invalid or missing label for {key}")
        pairs.append((a[key]["label"], b[key]["label"]))
    agreement = sum(x == y for x, y in pairs) / len(pairs)
    labels = sorted(allowed)
    pa = {label: sum(x == label for x, _ in pairs) / len(pairs) for label in labels}
    pb = {label: sum(y == label for _, y in pairs) / len(pairs) for label in labels}
    expected = sum(pa[label] * pb[label] for label in labels)
    kappa = (agreement - expected) / (1 - expected) if expected != 1 else 1.0
    result = {
        "n_trials": len(pairs), "observed_agreement": agreement, "cohen_kappa": kappa,
        "agreements": sum(x == y for x, y in pairs),
        "disagreements": sum(x != y for x, y in pairs),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("output", type=Path)
    ana = sub.add_parser("analyze")
    ana.add_argument("reviewer_a", type=Path)
    ana.add_argument("reviewer_b", type=Path)
    ana.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.output) if args.command == "export" else analyze(
        args.reviewer_a, args.reviewer_b, args.output
    )


if __name__ == "__main__":
    main()
