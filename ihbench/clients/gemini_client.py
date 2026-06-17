"""Google Gemini API client."""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import (
    APIClient,
    GenerationConfig,
    GenericMessage,
    is_audio_content,
    load_audio_bytes,
)


class GeminiChatClient(APIClient):
    """Client for Google Gemini API with audio support.

    Supports gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview, etc.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.thinking_budget = thinking_budget

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        from google.genai import types

        config = config or GenerationConfig()

        # Build contents list (Gemini uses a flat list of Content objects)
        contents = []
        for msg in messages:
            if msg.type == "audio" and is_audio_content(msg.content):
                wav_bytes = load_audio_bytes(msg.content)
                parts = [types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")]
            else:
                parts = [types.Part.from_text(text=msg.content)]
            contents.append(types.Content(role=msg.role if msg.role != "assistant" else "model", parts=parts))

        gen_config_kwargs: dict[str, Any] = {
            "system_instruction": system_message,
            "max_output_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if self.thinking_budget is not None:
            if self.thinking_budget == 0:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0,
                )
            else:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=self.thinking_budget,
                )

        generation_config = types.GenerateContentConfig(**gen_config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generation_config,
        )

        # Extract text from response, skipping thinking parts
        text_parts = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    continue
                if part.text:
                    text_parts.append(part.text)
        return "".join(text_parts)
