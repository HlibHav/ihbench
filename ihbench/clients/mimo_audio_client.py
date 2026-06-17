"""Client for MiMo-Audio-7B-Instruct.

Model card: https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct
Reference repo: https://github.com/XiaomiMiMo/MiMo-Audio

Modality support:
  - audio-in:  yes  (file paths; processed via MiMo's `preprocess_input`)
  - text-in:   yes  (text user turns)
  - text-out:  yes  (forward(..., task_name="spoken_dialogue") returns
                     only the text channel of the dual-channel decoder)

Implementation notes:
  * MiMo-Audio is NOT a standard HF AutoModel. Inference goes through:
        from src.mimo_audio.mimo_audio import MimoAudio
        model = MimoAudio(model_path, tokenizer_path)
    The repo has no setup.py / pyproject.toml — clone it once and the
    client adds it to sys.path. Tokenizer is a SEPARATE on-disk snapshot
    (`MiMo-Audio-Tokenizer`, ~1 GB).
  * MiMo's high-level chat methods don't fit IHBench:
      - `speech2text_dialogue_sft_multiturn`: assumes every user turn is
        audio (calls `preprocess_input(content)` on every user message,
        which `.reshape()`s — strings crash with "no attribute reshape").
        Also has no `system_prompt` kwarg.
      - `spoken_dialogue_sft_multiturn`: has system_prompt= but assistant
        turns must be `{"text", "audio"}` dicts (we only have text).
      - `text_dialogue_sft_multiturn`: pure text, can't carry audio.
    So we build the prompt directly with `InputSegment`s, mirroring
    `get_s2t_dialogue_sft_multiturn_prompt` but adding (a) a leading
    system turn and (b) per-turn dispatch on whether content is audio
    or text. Then call `model.forward(...)` with the same stopping
    criteria the high-level methods use.
  * `thinking` defaults to False (we emit `<think>\n\n</think>\n` to skip
    the thinking block, matching the model card's no-think behavior).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


_TOKENIZER_SEARCH_PATHS = [
    "XiaomiMiMo/MiMo-Audio-Tokenizer",
]


def _resolve_repo_path() -> str:
    env = os.environ.get("MIMO_AUDIO_REPO")
    if env and Path(env, "src", "mimo_audio", "mimo_audio.py").exists():
        return env
    raise RuntimeError(
        "MiMo-Audio repo not found. The repo has no setup.py and is not\n"
        "pip-installable. Clone it once and point MIMO_AUDIO_REPO at it:\n\n"
        "  git clone https://github.com/XiaomiMiMo/MiMo-Audio.git /path/to/MiMo-Audio\n"
        "  export MIMO_AUDIO_REPO=/path/to/MiMo-Audio\n"
    )


def _resolve_tokenizer_path() -> str:
    env = os.environ.get("MIMO_AUDIO_TOKENIZER")
    if env:
        return env
    for p in _TOKENIZER_SEARCH_PATHS:
        if Path(p).exists():
            return p
    return _TOKENIZER_SEARCH_PATHS[-1]


@register_client("XiaomiMiMo/MiMo-Audio-*")
@register_client("*/MiMo-Audio-7B-*")
class MimoAudioClient(HFLocalClient):
    """MiMo-Audio family client (7B-Instruct)."""

    def __init__(self, model_name_or_path: str, thinking: bool = False, **kwargs: Any):
        super().__init__(model_name_or_path, **kwargs)
        self._thinking = thinking

    def _load_model(self) -> None:
        if self._model is not None:
            return

        repo = _resolve_repo_path()
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from src.mimo_audio.mimo_audio import MimoAudio  # noqa: PLC0415

        tokenizer_path = _resolve_tokenizer_path()
        self._model = MimoAudio(self.model_name_or_path, tokenizer_path)
        self._repo_path = repo

    # ------------------------------------------------------------------
    # Direct prompt construction (bypassing MiMo's high-level chat APIs
    # because they don't support system + mixed user-modality turns).
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> tuple[Any, bool]:
        """Return (input_ids, any_audio).

        Mirrors `get_s2t_dialogue_sft_multiturn_prompt` from MiMo's repo,
        plus a leading system turn and per-message audio/text dispatch.
        """
        # Imports deferred until after sys.path insertion in _load_model.
        from src.mimo_audio.process_speechdata import InputSegment  # noqa: PLC0415

        zeroemb = self._model.speech_zeroemb_idx
        empty = self._model.empty_token

        def text_seg(text: str) -> "InputSegment":
            return InputSegment(text=text, speech_zeroemb_idx=zeroemb, text_zeroemb_idx=empty)

        def audio_seg(audio_path: str) -> "InputSegment":
            audio_tok = self._model.preprocess_input(audio_path)
            return InputSegment(audio=audio_tok, speech_zeroemb_idx=zeroemb, text_zeroemb_idx=empty)

        lm_prompt: list = []
        if system_message:
            lm_prompt += [
                text_seg("<|im_start|>system\n"),
                text_seg(system_message),
                text_seg("<|im_end|>\n"),
            ]

        any_audio = False
        for msg in messages:
            role_open = f"<|im_start|>{msg.role}\n"
            role_close = "<|im_end|>\n"
            if msg.type == "audio" and not self.text_only:
                lm_prompt += [
                    text_seg(role_open),
                    audio_seg(str(msg.content)),
                    text_seg(role_close),
                ]
                any_audio = True
            else:
                lm_prompt += [
                    text_seg(role_open),
                    text_seg(str(msg.content)),
                    text_seg(role_close),
                ]

        # Assistant generation prefix with (no-)thinking block.
        lm_prompt.append(text_seg("<|im_start|>assistant\n"))
        if not self._thinking:
            lm_prompt.append(text_seg("<think>\n\n</think>\n"))
        else:
            lm_prompt.append(text_seg("<think>\n"))

        input_ids = self._model.get_input_ids(lm_prompt)
        return input_ids, any_audio

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        self._load_model()
        # MiMo's `forward` uses the model's bundled generation config;
        # temperature/top_p from `config` aren't exposed. Match the parent
        # interface and silently ignore.
        del config

        from src.mimo_audio.modeling_mimo_audio import MiMoStopper  # noqa: PLC0415

        input_ids, any_audio = self._build_prompt(system_message, messages)
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self._model.tokenizer.eos_token_id, self._model.im_end_idx],
                group_size=self._model.group_size,
                audio_channels=self._model.audio_channels,
            )
        ]
        # task_name="spoken_dialogue" is what the audio-in chat methods use;
        # text_only mode falls through to "text_chat" since there are no
        # audio tokens to decode. Both return the text-channel string.
        task = "spoken_dialogue" if any_audio else "text_chat"
        # Suppress MiMo's noisy `print("Text channel:\t", ...)` inside
        # forward() — it dumps every generated response to stdout, which
        # is ~1 MB of log per epoch.
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415
        with contextlib.redirect_stdout(io.StringIO()):
            text = self._model.forward(
                input_ids,
                stopping_criteria=stopping_criteria,
                task_name=task,
                add_history=False,
            )
        # MiMo's `forward` returns the model's text-channel output. Two
        # cleanup passes:
        #   1. Strip the model's <think>...</think> reasoning block if
        #      thinking=True. We pre-fill "<think>\n" in the prefix, so
        #      the model emits the closing "</think>\n" tag once reasoning
        #      ends. Everything before "</think>" is internal monologue
        #      that the judge should not see.
        #   2. Strip trailing EOS / stopping-token residuals.
        if not isinstance(text, str):
            text = str(text)
        if self._thinking and "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.split("<|im_end|>")[0].strip()
