"""Generate the canonical v3 report from a completed live artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell(metric: dict) -> str:
    lo, hi = metric["ci_95"]
    return (
        f"{100 * metric['rate']:.1f}% ({100 * lo:.1f}%–{100 * hi:.1f}%), "
        f"{metric['events']}/{metric['n_trials']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text())
    if data.get("result_mode") != "live_api" or "completed_at_utc" not in data:
        raise SystemExit("Completed live artifact required")
    lines = [
        "# Adaptive Guardrail Robustness — Preregistered v3",
        "",
        f"Protocol: `{data['protocol']}`. Model: `{data['model']}`. "
        f"Dataset fingerprint: `{data['dataset_fingerprint']}`.",
        "",
        "| Defense | Adaptive GBR (Wilson 95% CI) | FPR (Wilson 95% CI) | Provider refusals |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("hardened", "hardened_v3"):
        metrics = data["configurations"][mode]["metrics"]
        lines.append(
            f"| `{mode}` | {_cell(metrics['adaptive_gbr'])} | {_cell(metrics['fpr'])} "
            f"| {metrics['provider_refusals']} |"
        )
    lines += ["", "## Paired tests", ""]
    for name, test in data["paired_tests"].items():
        lines.append(
            f"- `{name}`: exact McNemar p={test['p_value_two_sided']:.4f}; "
            f"old-only={test['baseline_only']}, v3-only={test['hardened_only']}, "
            f"N_pairs={test['n_pairs']}."
        )
    old = {r["id"]: r for r in data["configurations"]["hardened"]["rows"]}
    new = {r["id"]: r for r in data["configurations"]["hardened_v3"]["rows"]}
    fixed = [
        new[key] for key in old
        if old[key]["split"] == "benign" and not old[key]["allow"] and new[key]["allow"]
    ]
    regressions = [
        new[key] for key in old
        if old[key]["split"] == "benign" and old[key]["allow"] and not new[key]["allow"]
    ]
    residual = [r for r in new.values() if r["split"] == "benign" and not r["allow"]]
    lines += ["", "## Concrete transitions", ""]
    for label, rows in (("benign refusal fixed", fixed[:3]), ("benign regression", regressions[:2]),
                        ("residual false positive", residual[:2])):
        for row in rows:
            excerpt = " ".join(row["text"].split())[:220]
            lines.append(
                f"- **{row['id']} — {label}:** “{excerpt}”  \n"
                f"  `{row['detector']}` / `{row['category_decision']}`: {row['reason']}"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Zero observed bypasses do not prove zero risk; the Wilson upper bound is "
        "the defensible population-risk statement.",
        "- Provider refusals are real blocked outcomes but are separated because they "
        "are not attributable to repository defenses.",
        "- Multi-turn cases are serialized transcripts, not a stateful interactive attack.",
        "- Human-adjudicated, cross-model, and temporal conclusions remain pending "
        "until their external evidence requirements are met.",
        "- This is inference-time evaluation, not model training.",
    ]
    output = args.output or args.artifact.with_suffix(".md")
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
