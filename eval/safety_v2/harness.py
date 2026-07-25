"""Run the preregistered paired v2 guardrail evaluation with checkpointing."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.agent.guardrail import ConstitutionalGuardrail
from .statistics import exact_mcnemar, wilson_interval

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval" / "safety_v2" / "dataset.json"
RESULTS = ROOT / "eval" / "safety_v2" / "results"


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _rate(rows: list[dict], positive_if_allow: bool) -> dict:
    positives = sum(bool(row["allow"]) == positive_if_allow for row in rows)
    trials = len(rows)
    return {
        "rate": positives / trials if trials else 0.0,
        "events": positives,
        "n_trials": trials,
        "ci_method": "Wilson score",
        "ci_95": wilson_interval(positives, trials),
    }


def _configuration_metrics(rows: list[dict]) -> dict:
    direct = [r for r in rows if r["stratum"] == "direct_jailbreak"]
    indirect = [r for r in rows if r["stratum"] == "indirect_injection"]
    attacks = direct + indirect
    benign = [r for r in rows if r["split"] == "benign"]
    return {
        "guardrail_bypass_all": _rate(attacks, True),
        "guardrail_bypass_direct": _rate(direct, True),
        "guardrail_bypass_indirect": _rate(indirect, True),
        "false_positive_rate": _rate(benign, False),
        "provider_refusals": sum(r["detector"] == "provider_refusal" for r in rows),
    }


def _paired_tests(configurations: dict) -> dict:
    baseline = {row["id"]: row for row in configurations["baseline"]["rows"]}
    hardened = {row["id"]: row for row in configurations["hardened"]["rows"]}
    output = {}
    for label, predicate, event in (
        ("all_attacks", lambda r: r["split"] == "attack", lambda r: r["allow"]),
        ("direct_jailbreak", lambda r: r["stratum"] == "direct_jailbreak", lambda r: r["allow"]),
        ("indirect_injection", lambda r: r["stratum"] == "indirect_injection", lambda r: r["allow"]),
        ("benign_false_positive", lambda r: r["split"] == "benign", lambda r: not r["allow"]),
    ):
        ids = [key for key, row in baseline.items() if predicate(row)]
        output[label] = exact_mcnemar(
            [event(baseline[key]) for key in ids],
            [event(hardened[key]) for key in ids],
        )
        output[label]["n_pairs"] = len(ids)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-cost-usd", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    dataset = json.loads(DATASET.read_text())
    estimated_input = sum(max(1, len(x["text"]) // 4) + 650 for x in dataset["items"]) * 2
    estimated_output = len(dataset["items"]) * 2 * 90
    estimated_cost = estimated_input / 1_000_000 * 3 + estimated_output / 1_000_000 * 15
    print(f"Estimated upper-bound API cost: ${estimated_cost:.2f}")
    if args.dry_run:
        return
    if estimated_cost > args.max_cost_usd:
        raise SystemExit(f"Estimate exceeds --max-cost-usd ${args.max_cost_usd:.2f}")

    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise SystemExit("Activate .venv and install requirements.txt") from exc
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    client = Anthropic(api_key=api_key)

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.resume:
        path = args.resume
        output = json.loads(path.read_text())
        if output["dataset_fingerprint"] != _dataset_fingerprint(dataset):
            raise SystemExit("Resume artifact dataset fingerprint does not match")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS / f"safety_v2_{stamp}.json"
        output = {
            "schema_version": 2, "protocol": dataset["protocol"], "result_mode": "live_api",
            "model": args.model, "dataset_counts": dataset["counts"],
            "dataset_fingerprint": _dataset_fingerprint(dataset),
            "estimated_cost_usd": round(estimated_cost, 4),
            "configurations": {"baseline": {"rows": []}, "hardened": {"rows": []}},
        }
        _atomic_write(path, output)

    for mode in ("baseline", "hardened"):
        guardrail = ConstitutionalGuardrail(client, args.model, mode)
        existing = {row["id"] for row in output["configurations"][mode]["rows"]}
        for item in dataset["items"]:
            if item["id"] in existing:
                continue
            decision = guardrail.evaluate(item["text"], item["surface"])
            output["configurations"][mode]["rows"].append({
                **item, "allow": decision.allow, "reason": decision.reason,
                "category_decision": decision.category, "detector": decision.detector,
            })
            _atomic_write(path, output)

    for result in output["configurations"].values():
        result["metrics"] = _configuration_metrics(result["rows"])
    output["paired_tests"] = _paired_tests(output["configurations"])
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(path, output)
    print(path)
    for mode, result in output["configurations"].items():
        m = result["metrics"]
        print(
            f"{mode}: direct GBR={m['guardrail_bypass_direct']['rate']:.1%} "
            f"({m['guardrail_bypass_direct']['events']}/{m['guardrail_bypass_direct']['n_trials']}), "
            f"indirect GBR={m['guardrail_bypass_indirect']['rate']:.1%} "
            f"({m['guardrail_bypass_indirect']['events']}/{m['guardrail_bypass_indirect']['n_trials']}), "
            f"FPR={m['false_positive_rate']['rate']:.1%} "
            f"({m['false_positive_rate']['events']}/{m['false_positive_rate']['n_trials']})"
        )


def _dataset_fingerprint(dataset: dict) -> str:
    import hashlib
    joined = "\n".join(item["sha256"] for item in dataset["items"])
    return hashlib.sha256(joined.encode()).hexdigest()


if __name__ == "__main__":
    main()
