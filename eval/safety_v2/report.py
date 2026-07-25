"""Generate the v2 research report entirely from a completed live artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _metric_cell(metric: dict) -> str:
    lo, hi = metric["ci_95"]
    return f"{_pct(metric['rate'])} ({_pct(lo)}–{_pct(hi)}), {metric['events']}/{metric['n_trials']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text())
    if data.get("result_mode") != "live_api" or "completed_at_utc" not in data:
        raise SystemExit("A completed live_api artifact is required")
    lines = [
        "# Constitutional Guardrail Robustness — Preregistered v2",
        "",
        f"Model: `{data['model']}`. Protocol: `{data['protocol']}`. "
        f"Dataset fingerprint: `{data['dataset_fingerprint']}`.",
        "",
        "GBR is guardrail-bypass rate (an attack allowed beyond the guardrail), not "
        "end-to-end harmful-generation ASR. Wilson 95% intervals are shown.",
        "",
        "| Configuration | Direct jailbreak GBR | Indirect injection GBR | Combined GBR | FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("baseline", "hardened"):
        m = data["configurations"][label]["metrics"]
        lines.append(
            f"| {label.title()} | {_metric_cell(m['guardrail_bypass_direct'])} | "
            f"{_metric_cell(m['guardrail_bypass_indirect'])} | "
            f"{_metric_cell(m['guardrail_bypass_all'])} | "
            f"{_metric_cell(m['false_positive_rate'])} |"
        )
    lines += ["", "## Paired inference", ""]
    for name, test in data["paired_tests"].items():
        lines.append(
            f"- `{name}`: exact McNemar p={test['p_value_two_sided']:.4f}; "
            f"baseline-only={test['baseline_only']}, hardened-only={test['hardened_only']}, "
            f"N_pairs={test['n_pairs']}."
        )

    before = {r["id"]: r for r in data["configurations"]["baseline"]["rows"]}
    after = {r["id"]: r for r in data["configurations"]["hardened"]["rows"]}
    fixed_attacks = [
        after[key] for key in before
        if before[key]["split"] == "attack" and before[key]["allow"] and not after[key]["allow"]
    ]
    residual_attacks = [r for r in after.values() if r["split"] == "attack" and r["allow"]]
    fixed_benign = [
        after[key] for key in before
        if before[key]["split"] == "benign" and not before[key]["allow"] and after[key]["allow"]
    ]
    false_positives = [r for r in after.values() if r["split"] == "benign" and not r["allow"]]
    lines += ["", "## Mechanism and failure examples", ""]
    examples = (
        [("attack fixed", r) for r in fixed_attacks[:2]]
        + [("residual bypass", r) for r in residual_attacks[:2]]
        + [("false positive", r) for r in false_positives[:2]]
    )
    for label, row in examples:
        excerpt = " ".join(row["text"].split())[:220]
        lines.append(
            f"- **{row['id']} — {label}:** “{excerpt}”  \n"
            f"  `{row['detector']}` / `{row['category_decision']}`: {row['reason']}"
        )
    lines += [
        "",
        f"Attack bypasses fixed: {len(fixed_attacks)}. Residual hardened bypasses: "
        f"{len(residual_attacks)}. Benign refusals fixed: {len(fixed_benign)}. "
        f"Residual hardened false positives: {len(false_positives)}.",
        "",
        "Provider refusals are reported in the machine-readable artifact and retained "
        "in intention-to-treat outcomes; they are not attributed to the regex detector.",
    ]
    output = args.output or args.artifact.with_suffix(".md")
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
