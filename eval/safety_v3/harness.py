"""Paired adaptive evaluation for hardened v2 versus quote-aware hardened v3."""

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
DATASET = ROOT / "eval" / "safety_v3" / "dataset.json"
RESULTS = ROOT / "eval" / "safety_v3" / "results"


def _fingerprint(dataset: dict) -> str:
    return hashlib.sha256("\n".join(x["sha256"] for x in dataset["items"]).encode()).hexdigest()


def _atomic(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def _rate(rows: list[dict], event) -> dict:
    events = sum(event(row) for row in rows)
    return {
        "events": events, "n_trials": len(rows),
        "rate": events / len(rows) if rows else 0.0,
        "ci_95": wilson_interval(events, len(rows)), "ci_method": "Wilson score",
    }


def _metrics(rows: list[dict]) -> dict:
    attacks = [r for r in rows if r["split"] == "attack"]
    benign = [r for r in rows if r["split"] == "benign"]
    families = sorted({r["family"] for r in attacks})
    return {
        "adaptive_gbr": _rate(attacks, lambda r: r["allow"]),
        "fpr": _rate(benign, lambda r: not r["allow"]),
        "by_family": {
            family: _rate([r for r in attacks if r["family"] == family], lambda r: r["allow"])
            for family in families
        },
        "provider_refusals": sum(r["detector"] == "provider_refusal" for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=3.0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    dataset = json.loads(DATASET.read_text())
    input_tokens = sum(len(x["text"]) // 4 + 750 for x in dataset["items"]) * 2
    output_tokens = len(dataset["items"]) * 2 * 100
    prices = {"anthropic": (3.0, 15.0), "openai": (0.25, 2.0)}
    input_price, output_price = prices[args.provider]
    estimate = input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price
    print(f"Estimated upper-bound API cost: ${estimate:.2f}")
    if args.dry_run:
        return
    if estimate > args.max_cost_usd:
        raise SystemExit("Estimated cost exceeds --max-cost-usd")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.provider == "anthropic":
        from anthropic import Anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        client = OpenAIClientAdapter(api_key=os.environ["OPENAI_API_KEY"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.resume:
        path, output = args.resume, json.loads(args.resume.read_text())
        if output["dataset_fingerprint"] != _fingerprint(dataset):
            raise SystemExit("Dataset fingerprint mismatch")
        if output.get("provider", "anthropic") != args.provider or output["model"] != args.model:
            raise SystemExit("Resume artifact provider/model mismatch")
    else:
        path = RESULTS / f"safety_v3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        output = {
            "schema_version": 3, "protocol": dataset["protocol"], "model": args.model,
            "provider": args.provider,
            "result_mode": "live_api", "dataset_fingerprint": _fingerprint(dataset),
            "estimated_cost_usd": round(estimate, 4),
            "configurations": {"hardened": {"rows": []}, "hardened_v3": {"rows": []}},
        }
        _atomic(path, output)
    for mode in ("hardened", "hardened_v3"):
        guardrail = ConstitutionalGuardrail(client, args.model, mode)
        existing = {r["id"] for r in output["configurations"][mode]["rows"]}
        for item in dataset["items"]:
            if item["id"] in existing:
                continue
            decision = guardrail.evaluate(item["text"], item["surface"])
            output["configurations"][mode]["rows"].append({
                **item, "allow": decision.allow, "reason": decision.reason,
                "category_decision": decision.category, "detector": decision.detector,
            })
            _atomic(path, output)
    for config in output["configurations"].values():
        config["metrics"] = _metrics(config["rows"])
    old = {r["id"]: r for r in output["configurations"]["hardened"]["rows"]}
    new = {r["id"]: r for r in output["configurations"]["hardened_v3"]["rows"]}
    for label, predicate, event in (
        ("adaptive_gbr", lambda r: r["split"] == "attack", lambda r: r["allow"]),
        ("fpr", lambda r: r["split"] == "benign", lambda r: not r["allow"]),
    ):
        ids = [key for key, row in old.items() if predicate(row)]
        output.setdefault("paired_tests", {})[label] = exact_mcnemar(
            [event(old[key]) for key in ids], [event(new[key]) for key in ids]
        )
        output["paired_tests"][label]["n_pairs"] = len(ids)
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic(path, output)
    print(path)


if __name__ == "__main__":
    main()
