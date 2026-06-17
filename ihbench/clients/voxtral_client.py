"""Client for Voxtral-Small-24B-2507 (and Mini variant).

Model card: https://huggingface.co/mistralai/Voxtral-Small-24B-2507

Modality support:
  - audio-in:  yes  (the processor's `apply_chat_template` loads audio paths
                     itself via mistral-common's audio backend)
  - text-in:   yes  (same processor; just send `{"type": "text", ...}` parts)
  - text-out:  yes  (model only emits text)

Implementation notes:
  * Model class is `VoxtralForConditionalGeneration` (NOT AutoModel*).
  * Unlike Qwen2-Audio / Phi-4 where audio is passed as a separate `audios=`
    kwarg, Voxtral's `apply_chat_template(conversation)` reads the audio
    files directly from the `{"type": "audio", "path": ...}` part and
    returns a fully populated BatchFeature ready for the model. No
    librosa / soundfile call from our side.
  * Card warns "system prompts are not yet supported" — confirmed at
    smoke time: mistral-common < v13 raises "Found system messages at
    indexes [0] and audio chunks at indexes [...]. This is not allowed
    prior to the tokenizer version 13." Workaround: prepend the system
    text into the first user turn (as a text-part for audio user turns,
    or by concatenating into the text content otherwise). If the
    conversation starts with an assistant turn we insert a synthetic
    leading user turn carrying just the system text.
  * Requires `transformers >= 4.54.0` and `mistral-common[audio] >= 1.8.5`.
  * 24B in bf16 (~48 GB) does NOT fit on a single A100-40GB —
    `gpus_per_replica: 2` is mandatory.
  * ONGOING_RUNS.md flags a ">5-turn bug" historically. Smoke at
    depth=5 / 11 turns ran clean, so the bug may have been fixed
    upstream (transformers >= 4.54 / mistral-common >= 1.8.5). If a
    full-scale run surfaces it again, document and patch here.
"""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


@register_client("mistralai/Voxtral-*")
@register_client("*/Voxtral-*")
class VoxtralClient(HFLocalClient):
    """Voxtral family client (Small-24B, Mini-3B)."""

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        kwargs = dict(
            torch_dtype=getattr(torch, self._torch_dtype),
            device_map=self._device_map,
        )
        try:
            self._model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name_or_path, attn_implementation="flash_attention_2", **kwargs,
            )
        except (ImportError, ValueError) as e:
            print(f"[voxtral] flash_attention_2 unavailable ({e}); falling back to sdpa.")
            self._model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name_or_path, attn_implementation="sdpa", **kwargs,
            )

    # ------------------------------------------------------------------
    # Conversation formatting
    # ------------------------------------------------------------------

    def _build_conversation(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> list[dict[str, Any]]:
        # Voxtral's mistral-common tokenizer (< v13) rejects conversations
        # that have BOTH a `role: system` entry AND audio chunks — it errors:
        # "Found system messages ... and audio chunks ... not allowed prior
        # to tokenizer version 13". The card flagged this ("System prompts:
        # not yet supported"). Workaround: prepend the system text into the
        # first user turn as plain text so the signal isn't dropped.
        sys_prefix = system_message.strip() if system_message else ""
        conversation: list[dict[str, Any]] = []
        sys_consumed = False
        for msg in messages:
            if msg.type == "audio" and not self.text_only:
                parts: list[dict[str, Any]] = []
                if sys_prefix and not sys_consumed and msg.role == "user":
                    parts.append({"type": "text", "text": sys_prefix})
                    sys_consumed = True
                parts.append({"type": "audio", "path": str(msg.content)})
                conversation.append({"role": msg.role, "content": parts})
            else:
                text = str(msg.content)
                if sys_prefix and not sys_consumed and msg.role == "user":
                    text = f"{sys_prefix}\n\n{text}"
                    sys_consumed = True
                conversation.append({"role": msg.role, "content": text})
        # Edge case: conversation starts with assistant turn (no user yet).
        # Stuff the system into a leading synthetic user turn so it's not lost.
        if sys_prefix and not sys_consumed:
            conversation.insert(0, {"role": "user", "content": sys_prefix})
        return conversation

    def _build_inputs(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> dict[str, Any]:
        self._load_model()

        conversation = self._build_conversation(system_message, messages)
        # apply_chat_template returns a BatchFeature with input_ids,
        # attention_mask, and the audio features already extracted from
        # the file paths declared in the conversation dicts.
        inputs = self._processor.apply_chat_template(conversation)
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
