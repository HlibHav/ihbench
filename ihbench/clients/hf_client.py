"""HuggingFace local inference client using Accelerate.

This is a base implementation that can be subclassed for specific model
architectures (Qwen2-Audio, Phi-4-Multimodal, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ihbench.clients.base import GenerationConfig, GenericMessage, LocalClient


class HFLocalClient(LocalClient):
    """Client for HuggingFace models with local inference.

    Uses transformers + accelerate for model loading and generation.
    Subclass and override ``_build_inputs`` for model-specific formatting.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        text_only: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name_or_path, **kwargs)
        self._model = None
        self._processor = None
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        # text_only=True forces the client to ignore audio messages and use
        # their associated transcript text instead. Audio-capable subclasses
        # branch on this in _build_inputs(); the default base client is
        # text-only already so the flag is a no-op here, but accepting it
        # keeps a uniform constructor signature across all clients.
        self.text_only = text_only

    def _load_model(self) -> None:
        """Lazy-load model and processor."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype = getattr(torch, self._torch_dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            device_map=self._device_map,
            trust_remote_code=True,
        )

    def _build_inputs(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> dict[str, Any]:
        """Build model inputs from conversation. Override for model-specific formatting.

        Default implementation uses the processor's chat template with text-only messages.
        Audio messages are transcribed to their text content if available.
        """
        self._load_model()

        chat = [{"role": "system", "content": system_message}]
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else "[audio]"
            chat.append({"role": msg.role, "content": content})

        text = self._processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(text, return_tensors="pt")
        return {k: v.to(self._model.device) for k, v in inputs.items()}

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
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
            )

        generated = outputs[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True)
