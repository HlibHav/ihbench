"""Google Gemini Live API client (gemini-3.1-flash-live-preview).

Uses the Live API's WebSocket-based streaming for audio input and text output
via output_audio_transcription.

Prior conversation turns are seeded via send_client_content() (requires
history_config.initial_history_in_client_content=True), and the final user
audio is streamed in chunks via send_realtime_input().

Reference: https://ai.google.dev/gemini-api/docs/live-api/capabilities
"""

from __future__ import annotations

import asyncio
import io
import os
from typing import Any

import numpy as np
import soundfile as sf

from ihbench.clients.base import (
    APIClient,
    GenerationConfig,
    GenericMessage,
    is_audio_content,
    load_audio_bytes,
)

# PCM chunk size in bytes (1024 samples * 2 bytes/sample at 16kHz ≈ 64ms)
_PCM_CHUNK_BYTES = 2048
# Delay between chunks to simulate real-time streaming
_CHUNK_DELAY_S = 0.032


def _load_audio_16k(content: str) -> np.ndarray:
    """Read a WAV (path or b64:// URI) -> float32 mono at 16kHz."""
    audio, sr = sf.read(io.BytesIO(load_audio_bytes(content)))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample
        n_target = int(len(audio) * 16000 / sr)
        audio = resample(audio, n_target).astype(np.float32)
    return audio.astype(np.float32)


def _wav_to_pcm16k_bytes(content: str) -> bytes:
    """Read a WAV from a path or b64:// URI and return raw PCM16 bytes at 16kHz.

    Used both for the streamed final turn and for seeding prior user turns as
    inline-audio history (mime audio/pcm;rate=16000 -- the only audio format the
    Live API's send_client_content history channel actually honors)."""
    pcm16 = (np.clip(_load_audio_16k(content), -1.0, 1.0) * 32767).astype(np.int16)
    return pcm16.tobytes()


class GeminiLiveClient(APIClient):
    """Client for Gemini Live API via WebSocket.

    Prior conversation turns are seeded via send_client_content(), and the
    final user audio is streamed via send_realtime_input(). Responses are
    returned as text via output_audio_transcription.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        include_thoughts: bool | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.include_thoughts = include_thoughts

    async def _generate_async(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=self.api_key or os.environ.get("GEMINI_API_KEY"),
        )

        # Disable server-side VAD and demarcate the user turn manually via
        # activity_start / activity_end. Without this, Gemini's Live VAD
        # interprets IHBench's user audio (especially forceful pushback /
        # impatient phrasings) as the user interrupting the model's audio
        # output — fires `server_content.interrupted=True` and stays
        # silent, producing no `output_transcription`. That's the 11.7%
        # empty-response rate on the gemini-3.1-flash-live runs. With VAD
        # off, the audio between activity_start and activity_end is
        # treated as one complete user turn and the model responds.
        live_config_kwargs: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "output_audio_transcription": types.AudioTranscriptionConfig(),
            "system_instruction": system_message,
            "history_config": types.HistoryConfig(initial_history_in_client_content=True),
            "realtime_input_config": types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
            ),
        }
        if self.include_thoughts is not None:
            live_config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=self.include_thoughts,
            )
        live_config = types.LiveConnectConfig(**live_config_kwargs)

        model = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"

        # Find the last audio message
        last_audio_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "audio" and is_audio_content(messages[i].content):
                last_audio_idx = i
                break

        # Build context turns (everything except the final audio message, which
        # is streamed below). IHBench/AMC tag EVERY user turn as audio, so prior
        # user turns must be seeded as inline-audio Parts here -- earlier this
        # loop did `if msg.type == "audio": continue`, which dropped every
        # non-final user turn and left the model with assistant turns and no
        # user side of the conversation (crushing memory / retention).
        context_turns = []
        for i, msg in enumerate(messages):
            if i == last_audio_idx:
                continue
            role = "model" if msg.role == "assistant" else "user"
            if msg.type == "audio" and is_audio_content(msg.content):
                # MUST be raw PCM (audio/pcm;rate=16000), NOT audio/wav: the Live
                # API's send_client_content channel silently DROPS wav-mime audio
                # history (verified by probe), but honors raw PCM -- same format
                # the realtime streaming channel uses. (The regular non-Live
                # Gemini API accepts audio/wav; Live does not.)
                part = types.Part(
                    inline_data=types.Blob(
                        data=_wav_to_pcm16k_bytes(msg.content),
                        mime_type="audio/pcm;rate=16000",
                    )
                )
            else:
                part = types.Part(text=msg.content)
            context_turns.append(types.Content(role=role, parts=[part]))

        async with client.aio.live.connect(model=model, config=live_config) as session:
            # Seed conversation history.
            #
            # `turn_complete=True` is required when combining
            # send_client_content with the VAD-disabled / manual-activity
            # flow below; passing `turn_complete=False` here causes the
            # subsequent `activity_start` to be rejected with WS 1007
            # "invalid argument" by the server. Sending turn_complete=True
            # with the LAST history role being model (assistant) does NOT
            # trigger generation — the model is waiting for a user turn.
            if context_turns:
                await session.send_client_content(
                    turns=context_turns,
                    turn_complete=True,
                )

            # Send final user audio or text
            if last_audio_idx is not None:
                pcm_bytes = _wav_to_pcm16k_bytes(messages[last_audio_idx].content)
                # Explicit user-turn boundaries (server-side VAD disabled
                # in live_config). Streaming the audio chunks between
                # activity_start and activity_end tells Gemini "this is
                # one complete user turn", not "the user is barging in".
                await session.send_realtime_input(activity_start=types.ActivityStart())
                for i in range(0, len(pcm_bytes), _PCM_CHUNK_BYTES):
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm_bytes[i:i + _PCM_CHUNK_BYTES],
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                    await asyncio.sleep(_CHUNK_DELAY_S)
                await session.send_realtime_input(activity_end=types.ActivityEnd())
            else:
                last_user = None
                for msg in reversed(messages):
                    if msg.role == "user":
                        last_user = msg.content
                        break
                if last_user:
                    await session.send_client_content(
                        turns=[types.Content(
                            role="user",
                            parts=[types.Part(text=last_user)],
                        )],
                        turn_complete=True,
                    )

            # Collect transcribed text response.
            #
            # We break on `generation_complete`, NOT `turn_complete`. Live
            # API semantics:
            #   - `generation_complete` fires right after the model emits
            #     its last token (and the matching last output_transcription
            #     chunk) — i.e. exactly when our text_parts is fully
            #     populated.
            #   - `turn_complete` fires SECONDS later (observed: ~14 s
            #     extra), after Gemini's session-level wait for a user
            #     reply/barge-in.
            # Waiting for `turn_complete` is what produced the production
            # empty-response rate (11.7% on gemini-3.1-flash-live) — any
            # outer cutoff (WebSocket idle, job wall-time, network
            # blip) lands between the captured text and `turn_complete`,
            # and the in-progress text_parts is discarded when the
            # coroutine is cancelled.
            text_parts: list[str] = []
            async for response in session.receive():
                sc = response.server_content
                if sc:
                    if sc.output_transcription and sc.output_transcription.text:
                        text_parts.append(sc.output_transcription.text)
                    if sc.generation_complete or sc.turn_complete:
                        break

        return "".join(text_parts)

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        config = config or GenerationConfig()
        return asyncio.run(self._generate_async(system_message, messages, config))
