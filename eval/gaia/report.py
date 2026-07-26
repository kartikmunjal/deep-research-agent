"""Generate sanitized aggregate reports from private official-GAIA artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from .scoring import summarize


def rate(metric: dict) -> str:
    lo, hi = metric["ci_95"]
    return (
        f'{metric["events"]}/{metric["n_trials"]} '
        f'({100 * metric["rate"]:.1f}%; {100 * lo:.1f}–{100 * hi:.1f}%)'
    )


def exact_mcnemar(old_only: int, new_only: int) -> float:
    discordant = old_only + new_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) * 0.5**discordant
        for value in range(0, min(old_only, new_only) + 1)
    )
    return min(1.0, 2 * tail)


def paired_difference_ci(
    live: list[bool], replay: list[bool], resamples: int = 10_000
) -> list[float]:
    rng = random.Random(20260726)
    n = len(live)
    values = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(
            sum(int(live[index]) - int(replay[index]) for index in indices) / n
        )
    values.sort()
    return [values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]]


def subgroup(rows: list[dict], has_attachment: bool) -> dict:
    selected = [row for row in rows if bool(row["has_attachment"]) is has_attachment]
    return summarize(selected)["accuracy"]


def usage(payload: dict) -> dict:
    input_tokens = sum(row["usage"]["input_tokens"] for row in payload["rows"])
    output_tokens = sum(row["usage"]["output_tokens"] for row in payload["rows"])
    # claude-sonnet-4-6 standard API list price, recorded in the report method.
    estimated_usd = input_tokens / 1_000_000 * 3 + output_tokens / 1_000_000 * 15
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_anthropic_usd": round(estimated_usd, 4),
    }


def aggregate(payload: dict) -> dict:
    tool_calls = Counter(
        event["tool"] for row in payload["rows"] for event in row["tool_trace"]
    )
    tool_errors = Counter(
        event["error"]
        for row in payload["rows"]
        for event in row["tool_trace"]
        if event.get("error")
    )
    return {
        "mode": payload["mode"],
        "model": payload["model"],
        "accuracy": payload["summary"]["accuracy"],
        "attachment_accuracy": subgroup(payload["rows"], True),
        "non_attachment_accuracy": subgroup(payload["rows"], False),
        "failure_modes": payload["summary"]["failure_modes"],
        "tool_calls": dict(sorted(tool_calls.items())),
        "tool_errors": dict(sorted(tool_errors.items())),
        "usage": usage(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.artifacts]
    fingerprints = {item["task_set_fingerprint"] for item in payloads}
    if len(fingerprints) != 1:
        raise SystemExit("Artifacts do not use the same official task set")
    if any(len(item["rows"]) != 53 for item in payloads):
        raise SystemExit("Public report requires complete 53-task Level-1 artifacts")
    if any(any(row["failure_mode"] == "provider_error" for row in item["rows"]) for item in payloads):
        raise SystemExit("Provider-error rows make a public benchmark report invalid")
    by_mode = {item["mode"]: item for item in payloads}
    if set(by_mode) != {"live", "replay"}:
        raise SystemExit("Exactly one live and one replay artifact are required")

    live_rows = {row["task_id"]: row for row in by_mode["live"]["rows"]}
    replay_rows = {row["task_id"]: row for row in by_mode["replay"]["rows"]}
    if set(live_rows) != set(replay_rows):
        raise SystemExit("Live and replay task identities differ")
    ordered = sorted(live_rows)
    live_correct = [bool(live_rows[key]["correct"]) for key in ordered]
    replay_correct = [bool(replay_rows[key]["correct"]) for key in ordered]
    live_only = sum(a and not b for a, b in zip(live_correct, replay_correct))
    replay_only = sum(b and not a for a, b in zip(live_correct, replay_correct))
    delta = sum(live_correct) / len(ordered) - sum(replay_correct) / len(ordered)
    paired = {
        "n_pairs": len(ordered),
        "live_only_correct": live_only,
        "replay_only_correct": replay_only,
        "discordant_pairs": live_only + replay_only,
        "accuracy_difference_live_minus_replay": delta,
        "difference_bootstrap_95ci": paired_difference_ci(live_correct, replay_correct),
        "bootstrap_resamples": 10_000,
        "exact_mcnemar_p_two_sided": exact_mcnemar(live_only, replay_only),
    }
    public = {
        "schema_version": 1,
        "benchmark": "gaia-benchmark/GAIA",
        "split": "2023_validation",
        "level": 1,
        "task_set_fingerprint": next(iter(fingerprints)),
        "result_mode": "live_api_and_frozen_replay",
        "runs": [aggregate(by_mode["live"]), aggregate(by_mode["replay"])],
        "paired_comparison": paired,
        "privacy": "No gated task content, answers, predictions, task IDs, or tool outputs.",
    }

    lines = [
        "# Official GAIA Level-1 Validation — Live Tools vs Frozen Replay",
        "",
        "Dataset: gated `gaia-benchmark/GAIA`, 2023 validation Level 1. No questions,",
        "reference answers, attachments, predictions, task IDs, or tool outputs are",
        "reproduced. Both runs use `claude-sonnet-4-6` at temperature 0.",
        f"Task-set fingerprint: `{public['task_set_fingerprint']}`.",
        "",
        "| Mode | Accuracy (Wilson 95% CI) | Attachment tasks | No-attachment tasks |",
        "|---|---:|---:|---:|",
    ]
    for run in public["runs"]:
        lines.append(
            f'| `{run["mode"]}` | {rate(run["accuracy"])} | '
            f'{rate(run["attachment_accuracy"])} | '
            f'{rate(run["non_attachment_accuracy"])} |'
        )
    lo, hi = paired["difference_bootstrap_95ci"]
    lines += [
        "",
        "## Paired comparison",
        "",
        (
            f"Live minus replay accuracy: {100 * delta:.1f} percentage points "
            f"(paired bootstrap 95% CI {100 * lo:.1f} to {100 * hi:.1f}; "
            f"N_pairs={paired['n_pairs']}, 10,000 resamples)."
        ),
        (
            f"Exact McNemar p={paired['exact_mcnemar_p_two_sided']:.4f}; "
            f"live-only correct={live_only}, replay-only correct={replay_only}, "
            f"discordant pairs={paired['discordant_pairs']}."
        ),
        "",
        "## Failure taxonomy",
        "",
    ]
    for run in public["runs"]:
        failures = run["failure_modes"]
        detail = ", ".join(f"`{name}`={count}" for name, count in failures.items()) or "none"
        lines.append(f'- `{run["mode"]}`: {detail}.')
    lines += ["", "## Tool execution and estimated model cost", ""]
    for run in public["runs"]:
        tools = ", ".join(f"`{name}`={count}" for name, count in run["tool_calls"].items())
        errors = ", ".join(f"`{name}`={count}" for name, count in run["tool_errors"].items()) or "none"
        use = run["usage"]
        lines.append(
            f'- `{run["mode"]}`: calls {tools}; tool errors {errors}; '
            f'{use["input_tokens"]:,} input tokens, {use["output_tokens"]:,} output '
            f'tokens, estimated Anthropic cost ${use["estimated_anthropic_usd"]:.2f}.'
        )
    lines += [
        "",
        "Cost uses the standard Sonnet 4.6 list price of $3/million input tokens and",
        "$15/million output tokens; Tavily and OpenAI audio-transcription charges are",
        "not included.",
        "",
        "## Interpretation",
        "",
        "`live` executes Tavily search, restricted arithmetic, local attachment readers,",
        "and approved OpenAI audio transcription. `replay` reuses exact recorded",
        "search/calculator observations when the model repeats the same call and re-reads",
        "attachments. The lower replay result is driven partly by replay misses when the",
        "model changes tool arguments despite temperature 0. Therefore replay is not an",
        "oracle reasoning ceiling. The experiment shows that exact-call fixture mocks can",
        "underestimate a live agent when tool selection itself is nondeterministic.",
        "",
        "The defensible real-world score is the live result. Published GPT-4 figures are",
        "context only unless split, level, model, tools, and scoring protocol match.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.write_text(json.dumps(public, indent=2))
    print(args.output)
    print(json_output)


if __name__ == "__main__":
    main()
