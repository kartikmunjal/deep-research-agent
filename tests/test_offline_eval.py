from __future__ import annotations

import json

from eval import harness


def test_offline_eval_runs_without_api_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)

    result = harness.run_eval(
        configs=["plan_verify"],
        task_ids=["F01", "U01"],
        offline=True,
    )

    assert result["result_mode"] == "offline_fixture"
    assert len(result["results"]) == 2
    assert all(r.get("scores_are_synthetic") is True for r in result["results"])

    out_file = tmp_path / f"{result['run_id']}.json"
    assert out_file.exists()

    payload = json.loads(out_file.read_text())
    assert payload["result_mode"] == "offline_fixture"
