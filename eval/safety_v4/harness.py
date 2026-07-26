"""Run paired hardened_v3 versus hardened_v4 on the held-out typo dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from eval.safety_v2.statistics import exact_mcnemar, wilson_interval
from src.agent.guardrail import ConstitutionalGuardrail
from src.agent.provider_adapters import OpenAIClientAdapter

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval" / "safety_v4" / "dataset.json"
RESULTS = ROOT / "eval" / "safety_v4" / "results"
MODES = ("hardened_v3", "hardened_v4")


def _fingerprint(data: dict) -> str:
    return hashlib.sha256("\n".join(x["sha256"] for x in data["items"]).encode()).hexdigest()


def _write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _rate(rows: list[dict], event) -> dict:
    events = sum(event(row) for row in rows)
    return {"events": events, "n_trials": len(rows), "rate": events / len(rows),
            "ci_95": wilson_interval(events, len(rows)), "ci_method": "Wilson score"}


def _metrics(rows: list[dict]) -> dict:
    attacks = [r for r in rows if r["split"] == "attack"]
    benign = [r for r in rows if r["split"] == "benign"]
    return {
        "gbr": _rate(attacks, lambda r: r["allow"]),
        "fpr": _rate(benign, lambda r: not r["allow"]),
        "by_family": {
            family: _rate([r for r in attacks if r["family"] == family], lambda r: r["allow"])
            for family in sorted({r["family"] for r in attacks})
        },
        "provider_no_text": sum(r["detector"] in {"provider_refusal", "provider_length"} for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-cost-usd", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    data = json.loads(DATASET.read_text())
    input_tokens = sum(len(x["text"]) // 4 + 800 for x in data["items"]) * 2
    output_tokens = len(data["items"]) * 2 * 120
    prices = {"anthropic": (3.0, 15.0), "openai": (0.25, 2.0)}
    estimate = input_tokens / 1e6 * prices[args.provider][0] + output_tokens / 1e6 * prices[args.provider][1]
    print(f"Estimated upper-bound API cost: ${estimate:.2f}")
    if args.dry_run:
        return
    if estimate > args.max_cost_usd:
        raise SystemExit("Estimate exceeds --max-cost-usd")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY missing")
        client = OpenAIClientAdapter(os.environ["OPENAI_API_KEY"])
    else:
        from anthropic import Anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY missing")
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.resume:
        path, output = args.resume, json.loads(args.resume.read_text())
        if output["dataset_fingerprint"] != _fingerprint(data):
            raise SystemExit("Fingerprint mismatch")
    else:
        path = RESULTS / f"safety_v4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        output = {
            "schema_version": 4, "protocol": data["protocol"], "provider": args.provider,
            "model": args.model, "result_mode": "live_api",
            "dataset_fingerprint": _fingerprint(data), "estimated_cost_usd": round(estimate, 4),
            "configurations": {mode: {"rows": []} for mode in MODES},
        }
        _write(path, output)
    for mode in MODES:
        guardrail = ConstitutionalGuardrail(client, args.model, mode)
        existing = {r["id"] for r in output["configurations"][mode]["rows"]}
        for item in data["items"]:
            if item["id"] in existing:
                continue
            decision = guardrail.evaluate(item["text"], item["surface"])
            output["configurations"][mode]["rows"].append({
                **item, "allow": decision.allow, "reason": decision.reason,
                "category_decision": decision.category, "detector": decision.detector,
            })
            _write(path, output)
    for config in output["configurations"].values():
        config["metrics"] = _metrics(config["rows"])
    old = {r["id"]: r for r in output["configurations"][MODES[0]]["rows"]}
    new = {r["id"]: r for r in output["configurations"][MODES[1]]["rows"]}
    output["paired_tests"] = {}
    for label, split, event in (("gbr", "attack", lambda r: r["allow"]),
                                ("fpr", "benign", lambda r: not r["allow"])):
        ids = [key for key, row in old.items() if row["split"] == split]
        output["paired_tests"][label] = exact_mcnemar(
            [event(old[key]) for key in ids], [event(new[key]) for key in ids]
        )
        output["paired_tests"][label]["n_pairs"] = len(ids)
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(path, output)
    print(path)


if __name__ == "__main__":
    main()
