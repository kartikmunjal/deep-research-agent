from types import SimpleNamespace

from src.agent.provider_adapters import _OpenAIMessages


def test_openai_adapter_maps_response_to_guardrail_shape():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"allow": true, "reason": "safe", "category": "safe"}'
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
    )

    class Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == "snapshot"
            assert kwargs["response_format"] == {"type": "json_object"}
            assert kwargs["reasoning_effort"] == "minimal"
            assert kwargs["max_completion_tokens"] == 1024
            return completion

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    response = _OpenAIMessages(client).create(
        model="snapshot", max_tokens=160, messages=[{"role": "user", "content": "test"}]
    )
    assert response.content[0].text.startswith('{"allow"')
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 8
