"""Generate a human-readable, data-derived report from a safety eval artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--allow-offline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text())
    if data["result_mode"] != "live_api" and not args.allow_offline:
        raise SystemExit("Refusing performance report from non-live artifact; pass --allow-offline for QA only.")

    baseline = data["configurations"]["baseline"]
    hardened = data["configurations"]["hardened"]
    lines = [
        "# Guardrail Robustness Evaluation",
        "",
        f"Result mode: `{data['result_mode']}`. Model: `{data['model']}`.",
        "",
        "| Configuration | ASR (95% CI) | N_trials | FPR (95% CI) | N_trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in (("Baseline", baseline), ("Hardened", hardened)):
        m = result["metrics"]
        lines.append(
            f"| {name} | {_pct(m['asr'])} ({_pct(m['asr_95ci'][0])}–{_pct(m['asr_95ci'][1])}) "
            f"| {m['asr_n_trials']} | {_pct(m['fpr'])} "
            f"({_pct(m['fpr_95ci'][0])}–{_pct(m['fpr_95ci'][1])}) | {m['fpr_n_trials']} |"
        )

    before = {row["id"]: row for row in baseline["rows"]}
    after = {row["id"]: row for row in hardened["rows"]}
    fixed = [
        (before[key], after[key]) for key in before
        if before[key]["split"] == "attack" and before[key]["allow"] and not after[key]["allow"]
    ]
    residual = [row for row in hardened["rows"] if row["split"] == "attack" and row["allow"]]
    false_positives = [row for row in hardened["rows"] if row["split"] == "benign" and not row["allow"]]

    lines += ["", "## Mechanism and concrete failures", ""]
    examples = fixed[:2] + [(row, row) for row in residual[:1]]
    for old, new in examples:
        status = "fixed" if old["allow"] and not new["allow"] else "residual failure"
        excerpt = " ".join(new["text"].split())[:220]
        lines += [
            f"- **{new['id']} — {status}:** “{excerpt}”",
            f"  Hardened decision: `{new['category']}` via `{new['detector']}` — {new['reason']}",
        ]
    if not examples:
        lines.append("- No attack transition or residual failure was present.")
    lines += [
        "",
        f"Hardened residual attacks: {len(residual)}. Hardened false positives: {len(false_positives)}.",
        "The regex path is intended to explain gains on indirect injections; LLM-only "
        "changes are attributable to the few-shot decision boundary.",
    ]
    output = args.output or args.artifact.with_suffix(".md")
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
