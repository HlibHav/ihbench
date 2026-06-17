"""Client for Phi-4-multimodal-instruct.

Model card: https://huggingface.co/microsoft/Phi-4-multimodal-instruct

Modality support:
  - audio-in:  yes  (audios=[(np.ndarray, sample_rate), ...] tuples)
  - text-in:   yes  (just don't pass audios; inline placeholders are skipped)
  - text-out:  yes  (model only emits text)

Key differences from the Qwen-Omni family:
  * Loaded via AutoModelForCausalLM with trust_remote_code=True (custom
    modeling code lives in the HF repo).
  * No `chat_template` on the processor — you have to build the prompt
    yourself by concatenating <|user|>...<|end|><|assistant|>... turns.
    Audio is inlined as <|audio_1|>, <|audio_2|>, ... placeholders that
    must be numbered in the order audios are passed to the processor.
  * Audio is passed as a list of (np.ndarray, sample_rate) tuples, NOT
    file paths. soundfile.read does the I/O.
  * Phi-specific attn-impl kwarg is `_attn_implementation=` (leading
    underscore) and the value must be "flash_attention_2" or "eager"
    ("sdpa" is not implemented in their custom modeling).
"""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


@register_client("microsoft/Phi-4-multimodal-*")
@register_client("*/Phi-4-multimodal-*")
class Phi4MultimodalClient(HFLocalClient):
    """Phi-4 multimodal client (audio-in, text-out)."""

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig as HFGenerationConfig

        self._processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, trust_remote_code=True,
        )

        # Phi-4 multimodal does NOT support flash_attention_2 (its custom
        # modeling code raises ValueError on FA2). Don't even attempt it —
        # just load with eager so we skip the failed-load noise. SDPA is
        # also unimplemented in Phi-4's custom modeling.
        kwargs = dict(
            torch_dtype=getattr(torch, self._torch_dtype),
            device_map=self._device_map,
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **kwargs)

        # Pre-load Phi's recommended generation config so .generate inherits
        # eos_token_id and any model-specific defaults.
        try:
            self._gen_config = HFGenerationConfig.from_pretrained(self.model_name_or_path)
        except Exception:  # noqa: BLE001 — keep the client resilient to missing config
            self._gen_config = None

    # ------------------------------------------------------------------
    # Prompt building (no chat_template; manual concatenation)
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> dict[str, Any]:
        """Build a Phi-4 prompt + audios list.

        Phi-4 has no chat_template on the processor, so we concatenate
        <|user|>...<|end|><|assistant|>...<|end|> tokens manually.
        Audio messages get an <|audio_N|> placeholder inlined into the
        prompt; the corresponding (ndarray, sr) tuple is appended to the
        audios list in matching order.

        System message is prepended as <|system|>...<|end|>.
        """
        self._load_model()
        import soundfile as sf

        parts: list[str] = []
        if system_message:
            parts.append(f"<|system|>{system_message}<|end|>")

        audios: list[tuple] = []  # list of (np.ndarray, sample_rate)

        for msg in messages:
            role_tag = f"<|{msg.role}|>"
            if msg.type == "audio" and not self.text_only:
                # Read the WAV; soundfile returns (data, samplerate). Data is
                # float64 by default — Phi-4's processor accepts it directly.
                audio, sr = sf.read(str(msg.content))
                audios.append((audio, sr))
                # 1-indexed slot numbering per Phi-4's convention.
                slot = len(audios)
                parts.append(f"{role_tag}<|audio_{slot}|><|end|>")
            else:
                # Text-only path (either native text msg or text_only mode).
                parts.append(f"{role_tag}{str(msg.content)}<|end|>")

        # Assistant turn marker for generation continuation.
        parts.append("<|assistant|>")
        prompt = "".join(parts)

        # Pass audios kwarg only when we actually have audio; the processor
        # treats an empty list differently across versions.
        if audios:
            inputs = self._processor(text=prompt, audios=audios, return_tensors="pt")
        else:
            inputs = self._processor(text=prompt, return_tensors="pt")

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

        generate_kwargs = dict(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0,
        )
        # Use the model's shipped GenerationConfig if available (gives us
        # the right eos / pad / stopping criteria for Phi-4).
        if self._gen_config is not None:
            generate_kwargs["generation_config"] = self._gen_config

        with torch.inference_mode():
            out = self._model.generate(**inputs, **generate_kwargs)

        generated = out[:, input_len:]
        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""
