from __future__ import annotations

from dataclasses import dataclass

from src.agent.models import Evidence
from src.agent.pipeline import ResearchPipeline
from src.agent.planner import ResearchPlanner
from src.agent.verifier import ResearchVerifier


@dataclass
class _TextPart:
    text: str


@dataclass
class _LLMResponse:
    text: str

    @property
    def content(self) -> list[_TextPart]:
        return [_TextPart(self.text)]


class _FakeMessages:
    def __init__(self, outputs: list[str]):
        self._outputs = outputs

    def create(self, **_: object) -> _LLMResponse:
        return _LLMResponse(self._outputs.pop(0))


class _FakeClient:
    def __init__(self, outputs: list[str]):
        self.messages = _FakeMessages(outputs)


def test_planner_output_shape() -> None:
    client = _FakeClient(
        [
            '{"sub_questions": ["What is RLHF?", "How is it trained?", "What changed after RLHF?"], "reasoning": "Ordered from concept to impact."}'
        ]
    )
    planner = ResearchPlanner(client=client, model="fake-model")

    sub_questions, reasoning = planner.decompose("Explain RLHF impact on LLM alignment")

    assert 3 <= len(sub_questions) <= 5
    assert all(isinstance(sq, str) and sq for sq in sub_questions)
    assert isinstance(reasoning, str)


def test_verifier_returns_claim_level_judgments() -> None:
    client = _FakeClient(
        [
            '["RLHF became common after 2022 [1]", "RLHF removes all hallucinations [2]"]',
            '[{"claim": "RLHF became common after 2022 [1]", "verified": true, "supporting_excerpt": "Adoption increased in 2022."}, {"claim": "RLHF removes all hallucinations [2]", "verified": false, "supporting_excerpt": null}]',
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

    claims, unverified = verifier.verify("answer", evidence, sources=[])

    assert len(claims) == 2
    assert claims[0].verified is True
    assert claims[1].verified is False
    assert unverified == ["RLHF removes all hallucinations [2]"]


class _PlannerStub:
    def decompose(self, question: str) -> tuple[list[str], str]:
        return [f"sub-question: {question}"], "stub"


class _SearcherStub:
    def search(self, sub_question: str) -> tuple[list[Evidence], int]:
        return [
            Evidence(
                url="",
                title="",
                extracted_text="",
                sub_question=sub_question,
                search_successful=False,
            )
        ], 1


class _SynthesizerStub:
    def synthesize(
        self,
        question: str,
        sub_questions: list[str],
        evidence: list[Evidence],
    ) -> tuple[str, list[str], list[dict]]:
        return (
            "Insufficient evidence to answer all sub-questions.",
            sub_questions,
            [],
        )


class _VerifierStub:
    def verify(self, answer_text: str, evidence: list[Evidence], sources: list[dict]) -> tuple[list, list[str]]:
        raise AssertionError("Verifier should not run when skip_verification=True")


def test_pipeline_behavior_when_retrieval_fails() -> None:
    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.planner = _PlannerStub()
    pipeline.searcher = _SearcherStub()
    pipeline.synthesizer = _SynthesizerStub()
    pipeline.verifier = _VerifierStub()

    result = pipeline.run(
        "What are undisclosed training hyperparameters?",
        skip_planning=False,
        skip_verification=True,
        verbose=False,
    )

    assert len(result.unanswered_sub_questions) == 1
    assert result.completeness == 0.0
    assert result.tool_calls == 3
