"""Client for Qwen3-Omni-30B-A3B-Instruct (and same-family variants).

Model card: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct

Modality support:
  - audio-in:  yes  (via the qwen_omni_utils.process_mm_info helper)
  - text-in:   yes  (same processor; just send {"type": "text", ...} parts)
  - text-out:  yes  (we force this; the model also has a "talker" that can emit
                     audio at 24kHz — we disable it so only text comes out)

Implementation notes (worth knowing before editing):
  * The model class is `Qwen3OmniMoeForConditionalGeneration`, NOT
    AutoModelForCausalLM. Loading via Auto* picks the wrong head.
  * `model.disable_talker()` drops the audio-generation head and saves ~10 GB
    of GPU memory. We always call it for IHBench since we only consume text.
  * generate() returns a tuple `(text_ids, audio)` even when the talker is
    disabled — `audio` is just `None` in that case. The text portion is a
    GenerationOutput with `.sequences`, gated by
    thinker_return_dict_in_generate=True.
  * Output decoding slices past the prompt length, identical to what
    `HFLocalClient.generate` does, but the .sequences accessor differs — so
    we override `generate` rather than rely on the base implementation.
"""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


@register_client("Qwen/Qwen3-Omni-*")
@register_client("*/Qwen3-Omni-*")
class Qwen3OmniClient(HFLocalClient):
    """Qwen3-Omni family client (Instruct, Thinking, Captioner, etc.)."""

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        # Load processor first — it's cheap and we want both halves of the
        # client to fail together rather than leave _model populated and
        # _processor None on a partial failure.
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_name_or_path)

        # Pick attn impl: card recommends FA2; fall back to sdpa if FA2 isn't
        # available (driver mismatch, wheel missing, etc.).
        kwargs = dict(
            dtype=getattr(torch, self._torch_dtype),
            device_map=self._device_map,
        )
        try:
            self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_name_or_path, attn_implementation="flash_attention_2", **kwargs,
            )
        except (ImportError, ValueError) as e:
            print(f"[qwen3-omni] flash_attention_2 unavailable ({e}); falling back to sdpa.")
            self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_name_or_path, attn_implementation="sdpa", **kwargs,
            )

        # Drop the audio-generation head — we only consume text, and disabling
        # the talker saves ~10 GB and avoids returning audio tensors we'd
        # otherwise have to ignore.
        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()

    # ------------------------------------------------------------------
    # Conversation formatting
    # ------------------------------------------------------------------

    def _build_conversation(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> list[dict[str, Any]]:
        """Build the Qwen3-Omni chat dict structure.

        The system turn uses a single text part. User/assistant turns are
        lists of typed content parts. Audio messages become
        {"type": "audio", "audio": <path>}; the processor handles loading and
        resampling internally.
        """
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]}
        ]
        for msg in messages:
            parts: list[dict[str, Any]] = []
            if msg.type == "audio" and not self.text_only:
                # msg.content is the absolute path to a WAV on disk.
                parts.append({"type": "audio", "audio": str(msg.content)})
            else:
                # Either originally text, or text_only mode forcing the
                # transcript path. msg.content is always a str for text-typed
                # messages; for audio-typed messages in text_only mode the
                # caller is expected to have substituted the transcript
                # upstream (see run_inference_hf), so here we just stringify.
                parts.append({"type": "text", "text": str(msg.content)})
            conversation.append({"role": msg.role, "content": parts})
        return conversation

    def _build_inputs(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> dict[str, Any]:
        self._load_model()

        # qwen_omni_utils.process_mm_info collects the audio/image/video
        # buffers from the conversation dict so the processor can attach them
        # alongside the rendered chat-template text. This is the official
        # path; doing it manually is possible but the helper hides several
        # subtleties (sample rate, channel layout, URL fetching).
        from qwen_omni_utils import process_mm_info

        conversation = self._build_conversation(system_message, messages)
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        # BatchFeature.to() is dtype-aware: floats follow self._model.dtype,
        # integer tensors (input_ids, attention_mask) stay int. Casting via a
        # plain dict comprehension would force input_ids to bf16 and break
        # the embedding lookup.
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
            # Qwen3-Omni's generate() returns vary by flag combo. We pass
            # return_audio=False + thinker_return_dict_in_generate=True and
            # then normalize the return shape below, because the actual
            # observed shape depends on the transformers version and on
            # whether disable_talker() was called earlier. Empirically we've
            # seen:
            #   - (GenerationOutput, audio)
            #   - GenerationOutput alone (talker disabled)
            #   - a plain Tensor of generated ids (older builds)
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
                return_audio=False,
                thinker_return_dict_in_generate=True,
            )

        # Unwrap (first, audio) tuple shape if present.
        first = out[0] if isinstance(out, tuple) else out

        # The transformers Qwen3-Omni generate path has at least three
        # possible return shapes depending on version + talker state:
        #   (1) A plain str (or list[str]) — already-decoded output.
        #   (2) A GenerationOutput with .sequences (when
        #       thinker_return_dict_in_generate=True is honored).
        #   (3) A raw LongTensor of token ids (older builds).
        # Normalize all three down to a single string.
        if isinstance(first, str):
            return first
        if isinstance(first, list) and first and isinstance(first[0], str):
            return first[0]
        if hasattr(first, "sequences"):
            first = first.sequences

        generated = first[:, input_len:]
        decoded = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""
