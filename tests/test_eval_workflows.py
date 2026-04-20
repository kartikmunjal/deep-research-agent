from __future__ import annotations

import json

from eval import harness
from eval.benchmark_profiles import get_profile
from eval.measure_verifier_fpr import build_reference_examples
from eval.measure_replay_latency import summarize_latency
from eval.replay import freeze_run
from src.agent.models import Evidence
from src.agent.verifier import ResearchVerifier


class _TextPart:
    def __init__(self, text: str):
        self.text = text


class _LLMResponse:
    def __init__(self, text: str):
        self.content = [_TextPart(text)]


class _FakeMessages:
    def __init__(self, outputs: list[str]):
        self._outputs = outputs

    def create(self, **_: object) -> _LLMResponse:
        return _LLMResponse(self._outputs.pop(0))


class _FakeClient:
    def __init__(self, outputs: list[str]):
        self.messages = _FakeMessages(outputs)


def test_benchmark_profile_is_reproducible() -> None:
    profile = get_profile("core_live_28")

    assert profile.name == "core_live_28"
    assert len(profile.task_ids) == 28
    assert profile.task_ids[:3] == ("F01", "F02", "F03")
    assert profile.task_ids[-3:] == ("U06", "U07", "U08")


def test_offline_eval_accepts_named_profile(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)

    smoke = harness.run_eval(
        configs=["plan_verify"],
        profile="core_live_28",
        synthetic_smoke=True,
    )
    replay_file = tmp_path / "core_live_28.json"
    replay_file.write_text(
        json.dumps(
            freeze_run(smoke, benchmark_name="core_live_28", source_file="synthetic.json"),
            indent=2,
        )
    )
    result = harness.run_eval(
        configs=["plan_verify"],
        profile="core_live_28",
        offline=True,
        replay_benchmark=str(replay_file),
    )

    assert result["task_profile"] == "core_live_28"
    assert result["task_selection_note"]
    assert len(result["results"]) == 28
    assert result["result_mode"] == "replay_benchmark"


def test_verifier_can_check_explicit_claim_list() -> None:
    client = _FakeClient(
        [
            json.dumps(
                [
                    {
                        "claim": "RLHF became common after 2022 [1]",
                        "verified": True,
                        "supporting_excerpt": "Adoption increased in 2022.",
                    },
                    {
                        "claim": "RLHF removes all hallucinations [2]",
                        "verified": False,
                        "supporting_excerpt": None,
                    },
                ]
            )
        ]
    )
    verifier = ResearchVerifier(client=client, model="fake-model")
    evidence = [
        Evidence(
            url="https://example.com/a",
            title="Report A",
            extracted_text="Adoption increased in 2022.",
            sub_question="When did RLHF become common?",
            search_successful=True,
        )
    ]

    claims, unverified = verifier.verify_claims(
        [
            "RLHF became common after 2022 [1]",
            "RLHF removes all hallucinations [2]",
        ],
        evidence,
    )

    assert len(claims) == 2
    assert claims[0].verified is True
    assert claims[1].verified is False
    assert unverified == ["RLHF removes all hallucinations [2]"]


def test_build_reference_examples_uses_live_verified_claims() -> None:
    run = {
        "result_mode": "live_api",
        "results": [
            {
                "task_id": "F01",
                "question": "Question",
                "config": "plan_verify",
                "error": None,
                "evidence": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "extracted_text": "A supporting sentence.",
                        "sub_question": "SQ1",
                        "search_successful": True,
                    }
                ],
                "claims": [
                    {
                        "text": "Supported claim",
                        "verified": True,
                        "supporting_excerpt": "A supporting sentence.",
                    },
                    {
                        "text": "Unsupported claim",
                        "verified": False,
                        "supporting_excerpt": None,
                    },
                ],
            }
        ],
    }

    examples = build_reference_examples(run, sample_size=5, seed=1)

    assert len(examples) == 1
    assert examples[0]["original_claim"] == "Supported claim"
    assert examples[0]["paraphrased_claim"] == ""


def test_measure_replay_latency_uses_frozen_timing_fields() -> None:
    run = {
        "results": [
            {
                "task_id": "F01",
                "config": "plan_verify",
                "error": None,
                "stage_timings": {"search_seconds": 3.0},
                "sub_question_timings": [
                    {"sub_question": "sq1", "search_seconds": 2.0},
                    {"sub_question": "sq2", "search_seconds": 4.0},
                ],
            }
        ]
    }

    summary = summarize_latency(run)

    assert summary["tasks"] == 1
    assert summary["mean_async_search_seconds"] == 3.0
    assert summary["mean_serial_search_estimate_seconds"] == 6.0
    assert summary["mean_estimated_speedup"] == 2.0
