"""Client for Qwen2-Audio-7B-Instruct.

Model card: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

Modality support:
  - audio-in:  yes  (waveforms passed via `audios=` kwarg; chat template carries
                     `{"type": "audio", "audio_url": <path>}` placeholders)
  - text-in:   yes  (same processor; just send `{"type": "text", ...}` parts)
  - text-out:  yes  (model only emits text; no talker to disable)

Implementation notes:
  * Model class is `Qwen2AudioForConditionalGeneration` (NOT AutoModel*).
  * `audio_url` in the chat template is just a placeholder — the URL string
    itself is ignored at decode time. The real audio data is loaded with
    librosa and passed positionally via `processor(audios=...)` in the same
    order the placeholders appear. We can put any string in `audio_url`;
    we use the local file path for readability in case the template echoes
    it during debugging.
  * Sample rate comes from `processor.feature_extractor.sampling_rate`
    (16 kHz for Qwen2-Audio); librosa resamples on load.
"""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


@register_client("Qwen/Qwen2-Audio-*")
@register_client("*/Qwen2-Audio-*")
class Qwen2AudioClient(HFLocalClient):
    """Qwen2-Audio family client (7B-Instruct)."""

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        # flash_attention_2 has a runtime bug in Qwen2-Audio's Whisper encoder
        # (transformers/modeling_flash_attention_utils._upad_input → torch.gather
        # raises device-side assert "idx_dim < index_size"). The load itself
        # succeeds, so a try/except around from_pretrained doesn't catch it.
        # sdpa is the model card's default and works reliably.
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=getattr(torch, self._torch_dtype),
            device_map=self._device_map,
            attn_implementation="sdpa",
        )

    # ------------------------------------------------------------------
    # Conversation formatting
    # ------------------------------------------------------------------

    def _build_conversation(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Return (chat-template-conversation, ordered audio paths)."""
        conversation: list[dict[str, Any]] = []
        if system_message:
            conversation.append({"role": "system", "content": system_message})
        audio_paths: list[str] = []
        for msg in messages:
            if msg.type == "audio" and not self.text_only:
                path = str(msg.content)
                audio_paths.append(path)
                conversation.append({
                    "role": msg.role,
                    "content": [{"type": "audio", "audio_url": path}],
                })
            else:
                conversation.append({"role": msg.role, "content": str(msg.content)})
        return conversation, audio_paths

    def _build_inputs(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> dict[str, Any]:
        self._load_model()

        import librosa

        conversation, audio_paths = self._build_conversation(system_message, messages)
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )

        sr = self._processor.feature_extractor.sampling_rate
        audios = [librosa.load(p, sr=sr)[0] for p in audio_paths]

        # `audio=` is the post-4.52 kwarg; `audios=` is deprecated but still
        # accepted on 4.51. Pass `sampling_rate=sr` explicitly so the
        # WhisperFeatureExtractor doesn't fall back to its 16 kHz default
        # silently (it logs a warning otherwise — and would also produce
        # garbage features if the loaded waveform's rate differed).
        if audios:
            inputs = self._processor(
                text=text, audios=audios, return_tensors="pt", padding=True,
                sampling_rate=sr,
            )
        else:
            inputs = self._processor(text=text, return_tensors="pt", padding=True)

        # BatchFeature.to is dtype-aware (floats follow dtype, ints stay int).
        return inputs.to(self._model.device).to(self._model.dtype)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        import torch

        self._load_model()
        config = config or GenerationConfig()

        inputs = self._build_inputs(system_message, messages)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
            )

        generated = out[:, input_len:]
        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""
