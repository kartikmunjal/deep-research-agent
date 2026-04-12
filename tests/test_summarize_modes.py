from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import summarize_results


def _write_run(path: Path, run_id: str, mode: str) -> None:
    payload = {
        "run_id": run_id,
        "result_mode": mode,
        "results": [
            {
                "config": "plan_verify",
                "category": "factual",
                "tool_calls": 6,
                "scores": {
                    "citation_accuracy": 0.7,
                    "completeness": 0.6,
                    "hallucination_rate": 0.1,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload))


def test_latest_result_file_filters_by_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_run(tmp_path / "20260412T000100Z.json", "run-offline", "offline_fixture")
    _write_run(tmp_path / "20260412T000200Z.json", "run-live", "live_api")

    monkeypatch.setattr(summarize_results, "RESULTS_DIR", tmp_path)

    latest_any = summarize_results._latest_result_file(mode="any")
    latest_live = summarize_results._latest_result_file(mode="live_api")
    latest_offline = summarize_results._latest_result_file(mode="offline_fixture")

    assert latest_any.name == "20260412T000200Z.json"
    assert latest_live.name == "20260412T000200Z.json"
    assert latest_offline.name == "20260412T000100Z.json"


def test_latest_result_file_raises_if_mode_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_run(tmp_path / "20260412T000100Z.json", "run-offline", "offline_fixture")
    monkeypatch.setattr(summarize_results, "RESULTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        summarize_results._latest_result_file(mode="live_api")
