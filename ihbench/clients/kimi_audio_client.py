"""Client for Kimi-Audio-7B-Instruct.

Model card: https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
Reference repo: https://github.com/MoonshotAI/Kimi-Audio

Modality support:
  - audio-in:  yes  (file paths via `message_type="audio"`)
  - text-in:   **NOT SUPPORTED** at the model level. Kimi's
               `modeling_moonshot_kimia.forward` hard-indexes
               `whisper_input_feature[0].shape[-1]` (line ~681) and
               crashes with IndexError when no audio is present, even
               though the prompt_manager accepts `message_type="text"`
               messages. We therefore raise NotImplementedError in
               text_only mode so the §5.5 text-only ablation cleanly
               skips this model rather than wasting a SLURM run.
  - text-out:  yes  (`output_type="text"`; we never request audio out, so
                     the detokenizer is also disabled to save memory)

Implementation notes:
  * Kimi-Audio does NOT use the standard HF `AutoModel` + `AutoProcessor`
    pattern. Inference goes through:
        from kimia_infer.api.kimia import KimiAudio
        model = KimiAudio(model_path=..., load_detokenizer=False)
        wav, text = model.generate(messages, **sampling_params, output_type="text")
  * The repo has a pyproject.toml so it IS pip-installable from GitHub
    (declared in the YAML's `dependencies:` list as a git+https URL).
  * Messages use a non-standard schema:
        {"role": "user", "message_type": "audio", "content": "<wav_path>"}
        {"role": "user", "message_type": "text",  "content": "<text>"}
        {"role": "assistant", "message_type": "text", "content": "<text>"}
    There's also a combined `message_type="audio-text"` for assistants that
    speak AND output text, but we never need that path for IHBench.
  * Audio is passed as a path string. Kimi internally resamples (model
    operates at 24 kHz). IHBench WAVs at other rates should still work.
  * `output_type="text"` returns `(None, text)` (or `(empty_wav, text)`);
    we unpack defensively.
"""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


# Sampling defaults from the Kimi-Audio README. We mostly care about the
# text-channel temperature for our eval, but we keep the audio knobs
# populated because the API requires them as required kwargs.
_DEFAULT_SAMPLING_PARAMS = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}


@register_client("moonshotai/Kimi-Audio-*")
@register_client("*/Kimi-Audio-*")
class KimiAudioClient(HFLocalClient):
    """Kimi-Audio family client (7B-Instruct)."""

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from kimia_infer.api.kimia import KimiAudio

        # load_detokenizer=False skips the audio detokenizer (a separate
        # ~few-GB module). We only consume the text channel, so loading
        # the detokenizer is pure waste.
        self._model = KimiAudio(
            model_path=self.model_name_or_path,
            load_detokenizer=False,
        )

    # ------------------------------------------------------------------
    # Conversation formatting
    # ------------------------------------------------------------------

    def _build_message_list(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> list[dict[str, str]]:
        """Build Kimi-Audio's message dict list.

        Kimi's prompt_manager hard-asserts `role in ("user","assistant")`
        — a "system" entry raises bare AssertionError. Workaround: inject
        the system text as a synthetic leading user-text turn so the eval
        signal isn't dropped. Two adjacent user turns are fine (the
        prompt_manager just iterates messages in order).
        """
        out: list[dict[str, str]] = []
        if system_message:
            out.append({
                "role": "user",
                "message_type": "text",
                "content": system_message,
            })
        for msg in messages:
            if msg.type == "audio" and not self.text_only:
                out.append({
                    "role": msg.role,
                    "message_type": "audio",
                    "content": str(msg.content),
                })
            else:
                out.append({
                    "role": msg.role,
                    "message_type": "text",
                    "content": str(msg.content),
                })
        return out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        if self.text_only:
            raise NotImplementedError(
                "Kimi-Audio-7B does not support text-only inference: its forward "
                "hard-indexes whisper_input_feature[0] and crashes with IndexError "
                "when no audio is present. Skip this model in the §5.5 text-only "
                "ablation."
            )
        self._load_model()
        config = config or GenerationConfig()

        # The README's defaults pin text_temperature=0.0; let the caller
        # override via GenerationConfig.temperature when needed (e.g. for
        # diversity in multi-epoch CIs). audio_temperature stays at the
        # README default — we never sample audio.
        sampling = dict(_DEFAULT_SAMPLING_PARAMS)
        sampling["text_temperature"] = config.temperature

        message_list = self._build_message_list(system_message, messages)

        out = self._model.generate(
            message_list, **sampling, output_type="text",
        )
        # output_type="text" returns (wav_or_none, text). Some versions
        # return just text. Unwrap defensively.
        text: str = ""
        if isinstance(out, tuple):
            for item in out:
                if isinstance(item, str):
                    text = item
                    break
        elif isinstance(out, str):
            text = out

        # Kimi sometimes continues past EOS, emitting `<|im_msg_end|>` and
        # then unrelated tokens (we saw this on depth=0 IHBench samples).
        # Trim at the first message-end / im-end token to recover the
        # primary response.
        for tok in ("<|im_msg_end|>", "<|im_end|>"):
            if tok in text:
                text = text.split(tok, 1)[0]
        return text.strip()
