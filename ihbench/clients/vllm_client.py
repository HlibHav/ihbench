"""vLLM OpenAI-compatible API client."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from ihbench.clients.base import APIClient, GenerationConfig, GenericMessage


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    import wave

    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


class VLLMClient(APIClient):
    """Client for vLLM-served models via OpenAI-compatible API.

    Supports both text-only and audio-input models served through vLLM.
    For audio models, audio is passed as base64-encoded WAV in the request.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        served_model_name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.served_model_name = served_model_name or model_name

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        config = config or GenerationConfig()
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system_message}]

        for msg in messages:
            if msg.type == "audio" and isinstance(msg.content, np.ndarray):
                wav_bytes = _audio_to_wav_bytes(msg.content)
                b64 = base64.b64encode(wav_bytes).decode("utf-8")
                oai_messages.append({
                    "role": msg.role,
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    ],
                })
            else:
                oai_messages.append({"role": msg.role, "content": msg.content})

        resp = self.client.chat.completions.create(
            model=self.served_model_name,
            messages=oai_messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        return resp.choices[0].message.content or ""
