"""Generate a public aggregate report from private official-GAIA artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rate(metric: dict) -> str:
    lo, hi = metric["ci_95"]
    return (
        f'{metric["events"]}/{metric["n_trials"]} '
        f'({100 * metric["rate"]:.1f}%; {100 * lo:.1f}–{100 * hi:.1f}%)'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.artifacts]
    fingerprints = {item["task_set_fingerprint"] for item in payloads}
    if len(fingerprints) != 1:
        raise SystemExit("Artifacts do not use the same official task set")
    lines = [
        "# Official GAIA Validation Results",
        "",
        "Dataset: gated `gaia-benchmark/GAIA`, 2023 validation split. No questions,",
        "reference answers, attachments, predictions, or tool outputs are reproduced here.",
        f"Task-set fingerprint: `{next(iter(fingerprints))}`.",
        "",
        "| Mode | Model | Accuracy (Wilson 95% CI) | N_trials |",
        "|---|---|---:|---:|",
    ]
    for payload in payloads:
        metric = payload["summary"]["accuracy"]
        lines.append(
            f'| `{payload["mode"]}` | `{payload["model"]}` | {rate(metric)} | '
            f'{metric["n_trials"]} |'
        )
    lines += ["", "## Failure taxonomy", ""]
    for payload in payloads:
        failures = payload["summary"]["failure_modes"]
        detail = ", ".join(f"`{name}`={count}" for name, count in failures.items()) or "none"
        lines.append(f'- `{payload["mode"]}`: {detail}.')
    lines += [
        "",
        "## Interpretation",
        "",
        "`live` uses Tavily, the restricted calculator, and real attachment readers.",
        "`replay` reuses recorded search/calculator observations from the matching live",
        "run while re-reading the same local attachments. It controls tool/network",
        "variance; it is not an oracle and must not be called a pure reasoning ceiling.",
        "A live–replay gap therefore estimates execution/retrieval variance under this",
        "specific scaffold, not model reasoning ability in isolation.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
