"""IHBench client for Google Gemma 4 12B Unified (and family).

Gemma 4 12B Unified is an encoder-free multimodal LM: text, image, audio, and
video input, TEXT-ONLY output. IHBench uses text + audio only.

Capabilities:
  - audio-in:  yes  (16 kHz; the processor consumes audio referenced directly
                     in the chat template as `{"type": "audio", "audio": <path>}`
                     — no manual librosa load needed on our side).
  - text-in:   yes  (the `text_only` flag drops audio turns to their transcript
                     text, for the Sec 5.5 modality ablation).
  - text-out:  always (the model never emits audio).
  - thinking:  toggled via `enable_thinking` on `apply_chat_template`; the
               `thinking` constructor flag (YAML `client_kwargs: {thinking: ...}`)
               selects the no-think vs think variant, mirroring the Gemini /
               MiMo families. We always run `processor.parse_response()` on the
               decoded output so the judge sees only the final answer, never the
               raw `<think>...` reasoning trace.

Source of truth: the google/gemma-4-12B model card quick-start. Key API:
  from transformers import AutoProcessor, AutoModelForMultimodalLM
  processor.apply_chat_template(msgs, tokenize=True, return_dict=True,
      return_tensors="pt", add_generation_prompt=True, enable_thinking=<bool>)
  out = model.generate(**inputs, max_new_tokens=...)
  resp = processor.decode(out[0][input_len:], skip_special_tokens=False)
  processor.parse_response(resp)   # strips the thinking block

Notes:
  - Architecture is `Gemma4UnifiedForConditionalGeneration`; loaded via
    `AutoModelForMultimodalLM` (a Gemma-4-era transformers auto class), NOT
    AutoModelForCausalLM.
  - Recent transformers uses `dtype=` (not `torch_dtype=`) on from_pretrained.
  - Audio feature extractor (`Gemma4UnifiedAudioFeatureExtractor`) is 16 kHz.
  - 12B in bf16 (~24 GB) fits on a single A100-40GB -> gpus_per_replica: 1.
"""

from __future__ import annotations

from typing import Any

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.clients.hf_client import HFLocalClient
from ihbench.clients.registry import register_client


@register_client("google/gemma-4-12B*")
@register_client("*/gemma-4-12B*")
class Gemma4UnifiedClient(HFLocalClient):
    """Gemma 4 12B Unified: text+audio in, text out, optional thinking."""

    def __init__(
        self,
        model_name_or_path: str,
        thinking: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name_or_path, **kwargs)
        self._thinking = thinking

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        dtype = getattr(torch, self._torch_dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name_or_path,
            dtype=dtype,
            device_map=self._device_map,
        )

    def _build_conversation(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> list[dict[str, Any]]:
        """Gemma-4 chat-template conversation.

        Audio turns carry a `{"type": "audio", "audio": <path>}` content part
        (the processor loads the file itself). In text_only mode, audio turns
        fall back to their string content. Pure-text turns are emitted as a
        plain string content, matching the model card's text example.
        """
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": system_message}
        ]
        for msg in messages:
            if msg.type == "audio" and not self.text_only:
                # msg.content is a path on disk (IHBench passes audio by path).
                conversation.append(
                    {
                        "role": msg.role,
                        "content": [{"type": "audio", "audio": str(msg.content)}],
                    }
                )
            else:
                conversation.append({"role": msg.role, "content": str(msg.content)})
        return conversation

    def _build_inputs(
        self,
        system_message: str,
        messages: list[GenericMessage],
    ) -> dict[str, Any]:
        self._load_model()
        conversation = self._build_conversation(system_message, messages)
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=self._thinking,
        )
        # BatchFeature.to is dtype-aware (floats follow dtype, ints stay int).
        return inputs.to(self._model.device)

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
        # Keep special tokens so parse_response can locate/strip the thinking
        # block; then parse to the final answer text the judge should see.
        raw = self._processor.decode(generated, skip_special_tokens=False)
        parsed = self._processor.parse_response(raw)
        return self._strip_thinking(self._coerce_text(parsed))

    @staticmethod
    def _coerce_text(parsed: Any) -> str:
        """parse_response may return a str or a structured object; extract the
        final-answer text defensively (the exact return type is model-version
        dependent, so don't assume a shape)."""
        if isinstance(parsed, str):
            return parsed.strip()
        # dict-like with a common answer/content/text key
        for key in ("content", "answer", "text", "response"):
            if isinstance(parsed, dict) and key in parsed:
                return str(parsed[key]).strip()
        # object with a likely attribute
        for attr in ("content", "answer", "text"):
            if hasattr(parsed, attr):
                return str(getattr(parsed, attr)).strip()
        return str(parsed).strip()

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Defensively remove a Gemma-4 reasoning channel that survived
        parse_response.

        Gemma 4 emits reasoning as `<|channel>thought ... <channel|><answer><turn|>`.
        parse_response splits this only when enable_thinking was set; but the
        model sometimes emits a thought channel even in no-think mode, and a
        truncated generation can stop mid-thought before the `<channel|>`
        separator. In both cases the raw channel text leaks into `content`,
        which must never reach the judge. So:
          - if a `<channel|>` separator is present, keep only what follows the
            LAST one (the final answer), dropping the thought block;
          - if a `<|channel>thought` opener is present with NO separator, the
            answer was never generated (truncation) -> return empty so the
            runner records a failure instead of leaking the reasoning trace;
          - always strip a trailing `<turn|>` / leading/trailing whitespace.
        """
        if text is None:
            return ""
        t = text
        SEP = "<channel|>"
        OPEN = "<|channel>"
        if SEP in t:
            t = t.rsplit(SEP, 1)[1]
        elif OPEN in t:
            # opener but no separator: thought block, no final answer produced
            return ""
        # drop turn/end markers and any stray channel tokens
        for tok in ("<turn|>", "<|turn>", OPEN, SEP):
            t = t.replace(tok, "")
        return t.strip()
