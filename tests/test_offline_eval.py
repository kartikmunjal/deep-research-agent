from __future__ import annotations

import json

from eval import harness
from eval.replay import freeze_run


def test_offline_replay_runs_without_api_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)

    smoke = harness.run_eval(
        configs=["plan_verify"],
        task_ids=["F01", "U01"],
        synthetic_smoke=True,
    )
    frozen = freeze_run(smoke, benchmark_name="test_fixture", source_file="synthetic.json")
    replay_file = tmp_path / "test_fixture.json"
    replay_file.write_text(json.dumps(frozen, indent=2))

    result = harness.run_eval(
        configs=["plan_verify"],
        task_ids=["F01", "U01"],
        offline=True,
        replay_benchmark=str(replay_file),
    )

    assert result["result_mode"] == "replay_benchmark"
    assert len(result["results"]) == 2
    assert all(r.get("replayed_from_frozen_artifact") is True for r in result["results"])

    out_file = tmp_path / f"{result['run_id']}.json"
    assert out_file.exists()

    payload = json.loads(out_file.read_text())
    assert payload["result_mode"] == "replay_benchmark"
