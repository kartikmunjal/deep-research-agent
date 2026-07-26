"""Generate a cross-model v4 report entirely from completed live artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell(metric: dict) -> str:
    lo, hi = metric["ci_95"]
    return f"{metric['events']}/{metric['n_trials']} ({100*metric['rate']:.1f}%; {100*lo:.1f}–{100*hi:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) | {"artifact": str(path)} for path in args.artifacts]
    fingerprints = {run["dataset_fingerprint"] for run in runs}
    if len(fingerprints) != 1 or any("completed_at_utc" not in run for run in runs):
        raise SystemExit("Completed artifacts with one fingerprint required")
    lines = [
        "# Held-Out Typoglycemia Robustness — Preregistered v4",
        "",
        f"Protocol: `safety-v4-typoglycemia-holdout`. Dataset fingerprint: "
        f"`{next(iter(fingerprints))}`.",
        "",
        "| Provider / model | Defense | GBR (Wilson 95% CI) | FPR (Wilson 95% CI) | No-text |",
        "|---|---|---:|---:|---:|",
    ]
    for run in runs:
        for mode in ("hardened_v3", "hardened_v4"):
            metrics = run["configurations"][mode]["metrics"]
            lines.append(
                f"| {run['provider']} / `{run['model']}` | `{mode}` | {_cell(metrics['gbr'])} "
                f"| {_cell(metrics['fpr'])} | {metrics['provider_no_text']} |"
            )
    lines += ["", "## Paired inference", ""]
    for run in runs:
        for metric, test in run["paired_tests"].items():
            lines.append(
                f"- `{run['model']}` {metric}: p={test['p_value_two_sided']:.4f}; "
                f"v3-only={test['baseline_only']}, v4-only={test['hardened_only']}, "
                f"N={test['n_pairs']}."
            )
    lines += ["", "## Residual v4 bypass examples", ""]
    for run in runs:
        residual = [
            row for row in run["configurations"]["hardened_v4"]["rows"]
            if row["split"] == "attack" and row["allow"]
        ]
        for row in residual[:2]:
            lines.append(
                f"- **{run['model']} / {row['id']}:** “{' '.join(row['text'].split())[:200]}”  \n"
                f"  `{row['detector']}`: {row['reason']}"
            )
    lines += [
        "",
        "## Conclusion",
        "",
        "V4 is not promoted as the default defense. Across models it produced only "
        "small, statistically inconclusive GBR reductions; on OpenAI it also increased "
        "false positives. Internal-letter permutation remains the residual failure "
        "family. The correct engineering decision is to retain `hardened_v3` and treat "
        "the v4 normalization rule as a rejected experimental branch.",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
