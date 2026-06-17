"""Client for Qwen2.5-Omni-7B (and 3B variant).

Model card: https://huggingface.co/Qwen/Qwen2.5-Omni-7B

Modality support:
  - audio-in:  yes  (via qwen_omni_utils.process_mm_info)
  - text-in:   yes  (same processor; just send {"type": "text", ...} parts)
  - text-out:  yes  (we force this; the model also has a "talker" that emits
                     24 kHz audio — we disable it so only text comes out)

Implementation notes:
  * Model class is `Qwen2_5OmniForConditionalGeneration` (NOT AutoModel*).
  * Requires transformers built from the preview tag
    `v4.51.3-Qwen2.5-Omni-preview` — pinned in the YAML.
  * `model.disable_talker()` drops the audio head and saves ~2 GB.
  * generate() returns `(text_ids, audio)` when the talker is enabled,
    or `text_ids` alone when disabled. We normalise both.
  * Card recommends a specific system prompt for audio output; we always
    pass the eval's system message instead since we force text-out.
"""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


@register_client("Qwen/Qwen2.5-Omni-*")
@register_client("*/Qwen2.5-Omni-*")
class Qwen25OmniClient(HFLocalClient):
    """Qwen2.5-Omni family client (7B, 3B)."""

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name_or_path)

        # Workaround for the v4.51.3-Qwen2.5-Omni-preview transformers tag
        # (which reports as 4.52.0.dev0): when torch.distributed is
        # initialized, from_pretrained -> caching_allocator_warmup does
        #     re.compile("|".join([re.escape(p) for p in model._tp_plan]))
        # but Qwen2_5OmniForConditionalGeneration has _tp_plan = None,
        # raising "TypeError: 'NoneType' object is not iterable". Burned a
        # 5-min SLURM run (job 1229750) before catching this; set _tp_plan
        # to an empty dict on the class so the iteration is a no-op without
        # actually enabling tensor parallelism.
        if getattr(Qwen2_5OmniForConditionalGeneration, "_tp_plan", None) is None:
            Qwen2_5OmniForConditionalGeneration._tp_plan = {}

        # NOTE: the v4.51.3-Qwen2.5-Omni-preview transformers tag predates
        # the `dtype=` kwarg rename, so we must pass `torch_dtype=` here.
        # (Qwen3-Omni, which uses transformers main, takes `dtype=`.)
        kwargs = dict(
            torch_dtype=getattr(torch, self._torch_dtype),
            device_map=self._device_map,
        )
        try:
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_name_or_path, attn_implementation="flash_attention_2", **kwargs,
            )
        except (ImportError, ValueError) as e:
            print(f"[qwen2.5-omni] flash_attention_2 unavailable ({e}); falling back to sdpa.")
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_name_or_path, attn_implementation="sdpa", **kwargs,
            )

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
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]}
        ]
        for msg in messages:
            parts: list[dict[str, Any]] = []
            if msg.type == "audio" and not self.text_only:
                parts.append({"type": "audio", "audio": str(msg.content)})
            else:
                parts.append({"type": "text", "text": str(msg.content)})
            conversation.append({"role": msg.role, "content": parts})
        return conversation

    def _build_inputs(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> dict[str, Any]:
        self._load_model()

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
            # talker disabled -> return_audio=False; the function may still
            # return either a tuple (text_ids, None) or text_ids alone
            # depending on transformers version. Normalise below.
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
                use_audio_in_video=False,
                return_audio=False,
            )

        first = out[0] if isinstance(out, tuple) else out

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
