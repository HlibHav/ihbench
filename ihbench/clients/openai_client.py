"""OpenAI Chat Completions API client (GPT Audio, GPT Audio Mini, etc.)."""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import (
    APIClient,
    GenerationConfig,
    GenericMessage,
    is_audio_content,
    load_audio_b64,
)


class OpenAIChatClient(APIClient):
    """Client for OpenAI Chat Completions API with audio support.

    Supports gpt-audio, gpt-audio-mini, gpt-audio-1.5, and text-only
    models like gpt-4o, gpt-5, etc.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        config = config or GenerationConfig()
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system_message}]

        has_audio = False
        for msg in messages:
            if msg.type == "audio" and is_audio_content(msg.content):
                # Audio file path or b64:// URI
                b64 = load_audio_b64(msg.content)
                oai_messages.append({
                    "role": msg.role,
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    ],
                })
                has_audio = True
            else:
                oai_messages.append({"role": msg.role, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": oai_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_new_tokens,
        }
        # Audio models require explicit modalities
        if has_audio:
            kwargs["modalities"] = ["text"]
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
