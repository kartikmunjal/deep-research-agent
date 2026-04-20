"""Replay deterministic benchmark artifacts without API calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval.scoring import score_result
from src.agent.models import Claim, Evidence, ResearchAnswer


REPLAY_DIR = Path(__file__).parent / "replay"
DEFAULT_REPLAY_BENCHMARK = REPLAY_DIR / "core_live_28.json"


def resolve_replay_path(path: str | None = None) -> Path:
    """Resolve an explicit replay path or the default benchmark location."""
    if path:
        candidate = Path(path)
    else:
        candidate = DEFAULT_REPLAY_BENCHMARK

    if not candidate.exists():
        raise FileNotFoundError(
            "Replay benchmark not found. Expected frozen artifact at "
            f"{candidate}. Create one with `python -m eval.freeze_benchmark`."
        )

    return candidate


def load_replay_benchmark(path: str | None = None) -> dict:
    resolved = resolve_replay_path(path)
    with open(resolved) as f:
        payload = json.load(f)

    payload["_resolved_path"] = str(resolved)
    return payload


def freeze_run(
    run: dict,
    benchmark_name: str,
    source_file: str,
) -> dict:
    """Convert a timestamped eval run artifact into a replay benchmark artifact."""
    return {
        "format_version": "1.0",
        "benchmark_name": benchmark_name,
        "source_run_id": run.get("run_id"),
        "source_result_mode": run.get("result_mode"),
        "source_file": source_file,
        "source_git_commit": run.get("git_commit"),
        "source_model": run.get("model"),
        "task_profile": run.get("task_profile"),
        "task_selection_note": run.get("task_selection_note"),
        "benchmark_claims_allowed": run.get(
            "benchmark_claims_allowed",
            run.get("result_mode") == "live_api",
        ),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "configs": run.get("configs", []),
        "results": run.get("results", []),
    }


def replay_results(
    benchmark: dict,
    tasks: list[dict],
    configs: list[str],
    task_ids: list[str] | None = None,
) -> dict:
    """Deterministically reconstruct results and rescore them from frozen artifacts."""
    task_map = {task["id"]: task for task in tasks}
    allowed_task_ids = set(task_ids or [task["id"] for task in tasks])
    allowed_configs = set(configs)

    replayed_rows = []
    for row in benchmark.get("results", []):
        if row.get("task_id") not in allowed_task_ids:
            continue
        if row.get("config") not in allowed_configs:
            continue

        task = task_map.get(row["task_id"])
        if task is None:
            continue

        replayed_row = dict(row)
        replayed_row["scores_are_synthetic"] = False
        replayed_row["replayed_from_frozen_artifact"] = True

        if row.get("error"):
            replayed_row["scores"] = None
            replayed_rows.append(replayed_row)
            continue

        answer = _answer_from_row(row)
        scores = score_result(answer, task)
        source_scores = row.get("scores", {})
        if "cost_usd" in source_scores:
            scores["cost_usd"] = source_scores["cost_usd"]
        replayed_row["scores"] = scores
        replayed_rows.append(replayed_row)

    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": benchmark.get("source_git_commit", "unknown"),
        "model": benchmark.get("source_model", "frozen_replay"),
        "result_mode": "replay_benchmark",
        "benchmark_claims_allowed": benchmark.get("benchmark_claims_allowed", False),
        "task_profile": benchmark.get("task_profile"),
        "task_selection_note": benchmark.get("task_selection_note"),
        "replay_benchmark_name": benchmark.get("benchmark_name"),
        "replay_source_run_id": benchmark.get("source_run_id"),
        "replay_source_file": benchmark.get("_resolved_path", benchmark.get("source_file")),
        "configs": list(configs),
        "results": replayed_rows,
    }


def _answer_from_row(row: dict) -> ResearchAnswer:
    evidence = [Evidence(**item) for item in row.get("evidence", [])]
    claims = [Claim(**item) for item in row.get("claims", [])]
    return ResearchAnswer(
        question=row["question"],
        sub_questions=row.get("sub_questions", []),
        evidence=evidence,
        answer_text=row.get("answer_text", ""),
        sources=row.get("sources", []),
        claims=claims,
        unverified_claims=row.get("unverified_claims", []),
        unanswered_sub_questions=row.get("unanswered_sub_questions", []),
        tool_calls=row.get("tool_calls", 0),
        cost=None,
    )
