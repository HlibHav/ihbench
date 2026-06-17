"""OpenAI Realtime API client (GPT Realtime, GPT Realtime 1.5, GPT Realtime Mini).

Adapted from eval-runner/custom/inspectai/audio/wire.py::OpenAIRealtimeFormat.
"""

from __future__ import annotations

import asyncio
import base64
import io
import wave as _wave
from typing import Any

import numpy as np
from scipy.signal import resample

from ihbench.clients.base import (
    APIClient,
    GenerationConfig,
    GenericMessage,
    is_audio_content,
    load_audio_bytes,
)


def _wav_to_pcm24k_b64(content: str) -> str:
    """Read a WAV from a path or b64:// URI and return base64-encoded PCM16 at 24kHz."""
    with _wave.open(io.BytesIO(load_audio_bytes(content)), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    # Resample to 24 kHz
    if sr != 24000 and len(samples) > 0:
        n_target = int(len(samples) * 24000 / sr)
        samples = resample(samples, n_target).astype(np.float32)

    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode()


class OpenAIRealtimeClient(APIClient):
    """Client for OpenAI Realtime API via WebSocket.

    Audio is resampled from 16kHz WAV to 24kHz PCM16
    (the format the Realtime API expects).
    """

    def __init__(self, model_name: str, api_key: str | None = None, reasoning_effort: str | None = None, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort

    def _build_conversation_items(self, messages: list[GenericMessage]) -> list[dict]:
        """Convert GenericMessages into Realtime conversation items."""
        items: list[dict] = []
        for msg in messages:
            if msg.type == "audio" and is_audio_content(msg.content):
                pcm_b64 = _wav_to_pcm24k_b64(msg.content)
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": pcm_b64}],
                })
            elif msg.role == "user":
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.content}],
                })
            elif msg.role == "assistant":
                if msg.content:
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg.content}],
                    })
        return items

    async def _generate_async(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig,
    ) -> str:
        from openai import AsyncOpenAI

        oai = AsyncOpenAI(api_key=self.api_key)

        conversation_items = self._build_conversation_items(messages)

        async with oai.realtime.connect(model=self.model_name) as conn:
            # Wait for session.created
            async for event in conn:
                if event.type == "session.created":
                    break

            # Configure session
            session_config: dict[str, Any] = {
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": system_message,
            }
            if self.reasoning_effort:
                session_config["reasoning"] = {"effort": self.reasoning_effort}

            await conn.session.update(session=session_config)
            async for event in conn:
                if event.type == "session.updated":
                    break
                if event.type == "error":
                    raise RuntimeError(f"Realtime API error: {event.error}")

            # Inject conversation history
            n_items = len(conversation_items)
            for item in conversation_items:
                await conn.conversation.item.create(item=item)

            # Wait for all item confirmations
            done_count = 0
            if n_items > 0:
                async for event in conn:
                    if event.type == "conversation.item.done":
                        done_count += 1
                        if done_count >= n_items:
                            break
                    elif event.type == "error":
                        raise RuntimeError(f"Realtime API error: {event.error}")

            # Request a response
            await conn.response.create()

            # Collect text response
            text_parts: list[str] = []
            async for event in conn:
                if event.type == "response.output_text.delta":
                    text_parts.append(event.delta)
                elif event.type == "response.output_text.done":
                    pass
                elif event.type == "response.done":
                    break
                elif event.type == "error":
                    raise RuntimeError(f"Realtime API error: {event.error}")

        return "".join(text_parts)

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        config = config or GenerationConfig()
        return asyncio.run(self._generate_async(system_message, messages, config))
