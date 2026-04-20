"""Measure verifier rejection of supported paraphrased claims.

Workflow:
1. Export a template from a benchmark-claimable run:
   python -m eval.measure_verifier_fpr --run-file eval/results/<run>.json \
       --export-template eval/results/verifier_fpr_template.json
2. Fill in paraphrased_claim for each exported entry.
3. Measure verifier behavior on those supported paraphrases:
   python -m eval.measure_verifier_fpr --reference-file eval/results/verifier_fpr_template.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from src.agent.models import Evidence
from src.agent.verifier import ResearchVerifier


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_reference_examples(run: dict, sample_size: int, seed: int) -> list[dict]:
    """Sample verified claims from a benchmark-claimable run for paraphrase stress tests."""
    if not run.get("benchmark_claims_allowed", run.get("result_mode") == "live_api"):
        raise ValueError(
            "Reference examples must come from a benchmark-claimable live_api or replay_benchmark run."
        )

    candidates: list[dict] = []
    for row in run.get("results", []):
        if row.get("config") != "plan_verify":
            continue
        if row.get("error"):
            continue
        evidence = [
            {
                "url": ev.get("url", ""),
                "title": ev.get("title", ""),
                "extracted_text": ev.get("extracted_text", ""),
                "sub_question": ev.get("sub_question", ""),
                "search_successful": ev.get("search_successful", True),
            }
            for ev in row.get("evidence", [])
            if ev.get("search_successful", True) and ev.get("extracted_text")
        ]
        if not evidence:
            continue

        for claim in row.get("claims", []):
            if not claim.get("verified"):
                continue
            candidates.append(
                {
                    "task_id": row["task_id"],
                    "question": row["question"],
                    "original_claim": claim["text"],
                    "supporting_excerpt": claim.get("supporting_excerpt"),
                    "paraphrased_claim": "",
                    "expected_verified": True,
                    "evidence": evidence,
                }
            )

    if not candidates:
        raise ValueError("No verified claims found in the supplied live run.")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: min(sample_size, len(candidates))]


def _evidence_objects(entry: dict) -> list[Evidence]:
    return [
        Evidence(
            url=item["url"],
            title=item["title"],
            extracted_text=item["extracted_text"],
            sub_question=item["sub_question"],
            search_successful=item.get("search_successful", True),
        )
        for item in entry["evidence"]
    ]


def evaluate_reference_file(reference: dict, model: str) -> dict:
    """Run the verifier on manually paraphrased claims and compute rejection rate."""
    from anthropic import Anthropic

    api_key = os.environ["ANTHROPIC_API_KEY"]
    verifier = ResearchVerifier(Anthropic(api_key=api_key), model=model)

    rows = []
    for entry in reference.get("examples", []):
        paraphrase = entry.get("paraphrased_claim", "").strip()
        if not paraphrase:
            raise ValueError("Every reference example must include paraphrased_claim.")

        claims, _ = verifier.verify_claims([paraphrase], _evidence_objects(entry))
        verdict = claims[0].verified if claims else False
        rows.append(
            {
                "task_id": entry["task_id"],
                "original_claim": entry["original_claim"],
                "paraphrased_claim": paraphrase,
                "expected_verified": True,
                "verifier_verified": verdict,
                "supporting_excerpt": claims[0].supporting_excerpt if claims else None,
            }
        )

    total = len(rows)
    rejected = sum(1 for row in rows if not row["verifier_verified"])
    return {
        "metric_name": "supported_claim_false_rejection_rate",
        "total_examples": total,
        "false_rejections": rejected,
        "supported_claim_false_rejection_rate": round(rejected / total, 3) if total else 0.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure verifier rejection on supported paraphrases")
    parser.add_argument("--run-file", type=str, help="Live result artifact used to export examples")
    parser.add_argument("--reference-file", type=str, help="JSON file with paraphrased reference examples")
    parser.add_argument("--export-template", type=str, help="Where to write the sampled reference template")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--output", type=str, help="Optional JSON output path for measured results")
    args = parser.parse_args()

    if bool(args.run_file) == bool(args.reference_file):
        raise SystemExit("Use either --run-file or --reference-file.")

    if args.run_file:
        if not args.export_template:
            raise SystemExit("--export-template is required with --run-file.")
        run = _load(Path(args.run_file))
        examples = build_reference_examples(run, args.sample_size, args.seed)
        payload = {
            "source_run_id": run.get("run_id"),
            "source_file": args.run_file,
            "sample_size": len(examples),
            "instructions": (
                "Rewrite each original_claim into a semantically equivalent paraphrase "
                "in paraphrased_claim, then rerun this script with --reference-file."
            ),
            "examples": examples,
        }
        out_path = Path(args.export_template)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote reference template: {out_path}")
        return

    reference = _load(Path(args.reference_file))
    results = evaluate_reference_file(reference, args.model)
    print(f"Metric: {results['metric_name']}")
    print(f"Total examples: {results['total_examples']}")
    print(f"False rejections: {results['false_rejections']}")
    print(
        "Supported-claim false rejection rate: "
        f"{results['supported_claim_false_rejection_rate']:.1%}"
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results: {out_path}")


if __name__ == "__main__":
    main()
