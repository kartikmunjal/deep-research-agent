"""Regenerate the top-level official-GAIA findings from the public aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULT = ROOT / "eval" / "gaia" / "results" / "official_gaia_l1_20260726.json"
START = "## Official GAIA Validation Track"
END = "## Historical Self-Authored GAIA-Style Snapshot (Not GAIA)"


def rate(metric: dict) -> str:
    lo, hi = metric["ci_95"]
    return (
        f'{metric["events"]}/{metric["n_trials"]} '
        f'({100 * metric["rate"]:.1f}%; {100 * lo:.1f}–{100 * hi:.1f}%)'
    )


def build() -> str:
    data = json.loads(RESULT.read_text())
    runs = {run["mode"]: run for run in data["runs"]}
    paired = data["paired_comparison"]
    live_failures = runs["live"]["failure_modes"]
    lo, hi = paired["difference_bootstrap_95ci"]
    return "\n".join(
        [
            START,
            "",
            "The attachment-aware harness evaluates the gated official",
            "`gaia-benchmark/GAIA` 2023 Level-1 validation split with real Tavily",
            "search, restricted arithmetic, local document/image readers, and approved",
            "OpenAI audio transcription. Gated questions, answers, predictions, task IDs,",
            "attachments, and tool outputs remain private.",
            "",
            "| Mode | Accuracy (Wilson 95% CI) | Attachment tasks | No-attachment tasks |",
            "|---|---:|---:|---:|",
            (
                f"| Live tools | {rate(runs['live']['accuracy'])} | "
                f"{rate(runs['live']['attachment_accuracy'])} | "
                f"{rate(runs['live']['non_attachment_accuracy'])} |"
            ),
            (
                f"| Frozen exact-call replay | {rate(runs['replay']['accuracy'])} | "
                f"{rate(runs['replay']['attachment_accuracy'])} | "
                f"{rate(runs['replay']['non_attachment_accuracy'])} |"
            ),
            "",
            (
                f"Live exceeded replay by "
                f"{100 * paired['accuracy_difference_live_minus_replay']:.1f} percentage "
                f"points (paired bootstrap 95% CI {100 * lo:.1f}–{100 * hi:.1f}; "
                f"N_pairs={paired['n_pairs']}; exact McNemar "
                f"p={paired['exact_mcnemar_p_two_sided']:.4f}). "
                f"Live-only correct={paired['live_only_correct']}; "
                f"replay-only correct={paired['replay_only_correct']}."
            ),
            "",
            "This rejects the simplistic interpretation of mocked tools as a reasoning",
            "ceiling. Even at temperature 0, the model often reformulated tool arguments;",
            "exact-call replay then missed recorded observations. The defensible real-world",
            (
                "result is the live score. Live failures were "
                f"{live_failures.get('reasoning_error', 0)} reasoning errors and "
                f"{live_failures.get('tool_error', 0)} step-limit/tool errors; no "
                "provider errors remained in the completed artifact."
            ),
            "",
            "See the [preregistration](RESEARCH_PLAN_GAIA.md),",
            "[evaluation instructions](eval/gaia/README.md), and",
            "[sanitized full report](eval/gaia/results/official_gaia_l1_20260726.md).",
            "",
            "",
        ]
    )


def render(current: str) -> str:
    start = current.index(START)
    end = current.index(END, start)
    return current[:start] + build() + current[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = README.read_text()
    generated = render(current)
    if args.check:
        if generated != current:
            raise SystemExit("README GAIA findings are out of date")
        print("README GAIA findings are synchronized")
    else:
        README.write_text(generated)
        print(README)


if __name__ == "__main__":
    main()
