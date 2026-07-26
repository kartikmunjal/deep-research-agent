"""Provider adapters exposing the minimal messages.create interface used by guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class _TextBlock:
    text: str


class _OpenAIMessages:
    def __init__(self, client: object):
        self.client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        temperature: float = 0,
    ) -> object:
        _ = temperature  # GPT-5 reasoning snapshots do not use sampling temperature.
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = SimpleNamespace(
            input_tokens=getattr(response.usage, "prompt_tokens", 0),
            output_tokens=getattr(response.usage, "completion_tokens", 0),
        )
        return SimpleNamespace(
            content=[_TextBlock(text)] if text else [],
            usage=usage,
            stop_reason=str(choice.finish_reason or "unknown"),
        )


class OpenAIClientAdapter:
    """Adapt the OpenAI Chat Completions SDK to ConstitutionalGuardrail."""

    def __init__(self, api_key: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.messages = _OpenAIMessages(self._client)
