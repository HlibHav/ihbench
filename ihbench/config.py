"""YAML-based model and evaluation configuration.

Supports loading from YAML files or constructing from CLI arguments.
CLI arguments override YAML values when both are provided.

Example YAML (model config):

    backend: openai
    model: gpt-4o-audio-preview
    api_key: ${OPENAI_API_KEY}   # env var expansion
    temperature: 1.0
    max_new_tokens: 4096
    max_workers: 8

Example YAML (eval config):

    judge_backend: openai
    judge_model: gpt-4o
    judge_api_key: ${OPENAI_API_KEY}
    judge_temperature: 1.0
    judge_max_new_tokens: 16384
    judge_max_workers: 40
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} and $VAR patterns in string values."""
    if not isinstance(value, str):
        return value
    return re.sub(
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z_0-9]*)",
        lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
        value,
    )


def _expand_all(d: dict) -> dict:
    return {k: _expand_env(v) for k, v in d.items()}


@dataclass
class ModelConfig:
    """Configuration for a model client."""

    backend: str  # openai, openai-realtime, gemini, vllm, hf
    model: str
    api_key: str | None = None
    base_url: str | None = None
    served_model_name: str | None = None
    thinking_budget: int | None = None
    reasoning_effort: str | None = None  # For OpenAI reasoning models: "low", "medium", "high", "xhigh"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 4096
    max_workers: int = 8  # For API parallel calls

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        raw = _expand_all(raw)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_args(cls, args: Any) -> ModelConfig:
        """Build from argparse namespace. Only sets fields that are present in args."""
        kwargs = {}
        for fld in cls.__dataclass_fields__:
            val = getattr(args, fld, None)
            if val is not None:
                kwargs[fld] = val
        return cls(**kwargs)

    @classmethod
    def from_yaml_and_args(cls, yaml_path: str | Path | None, args: Any) -> ModelConfig:
        """Load from YAML, then override with any CLI args that were explicitly set."""
        if yaml_path:
            cfg = cls.from_yaml(yaml_path)
            # Override with non-None CLI args
            for fld in cls.__dataclass_fields__:
                val = getattr(args, fld, None)
                if val is not None:
                    setattr(cfg, fld, val)
            return cfg
        return cls.from_args(args)

    def make_client(self):
        """Instantiate the appropriate model client."""
        if self.backend == "openai":
            from ihbench.clients.openai_client import OpenAIChatClient

            return OpenAIChatClient(
                model_name=self.model, api_key=self.api_key,
                base_url=self.base_url, reasoning_effort=self.reasoning_effort,
            )
        elif self.backend == "openai-realtime":
            from ihbench.clients.openai_realtime_client import OpenAIRealtimeClient

            return OpenAIRealtimeClient(
                model_name=self.model, api_key=self.api_key,
                reasoning_effort=self.reasoning_effort,
            )
        elif self.backend == "gemini":
            from ihbench.clients.gemini_client import GeminiChatClient

            return GeminiChatClient(
                model_name=self.model,
                api_key=self.api_key,
                thinking_budget=self.thinking_budget,
            )
        elif self.backend == "vllm":
            from ihbench.clients.vllm_client import VLLMClient

            return VLLMClient(
                model_name=self.model,
                base_url=self.base_url or "http://localhost:8000/v1",
                served_model_name=self.served_model_name,
            )
        elif self.backend == "hf":
            from ihbench.clients.hf_client import HFLocalClient

            return HFLocalClient(
                model_name_or_path=self.model,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


@dataclass
class JudgeConfig:
    """Configuration for the judge model."""

    backend: str = "openai"
    model: str = "gpt-5.4-mini"
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None
    temperature: float = 1.0
    max_new_tokens: int = 16384
    max_workers: int = 40

    @classmethod
    def from_yaml(cls, path: str | Path) -> JudgeConfig:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        raw = _expand_all(raw)
        # Support both "judge_*" prefixed and unprefixed keys
        kwargs = {}
        for fld in cls.__dataclass_fields__:
            if fld in raw:
                kwargs[fld] = raw[fld]
            elif f"judge_{fld}" in raw:
                kwargs[fld] = raw[f"judge_{fld}"]
        return cls(**kwargs)

    def make_client(self):
        """Instantiate the judge client (always an API client)."""
        if self.backend == "openai":
            from ihbench.clients.openai_client import OpenAIChatClient

            return OpenAIChatClient(
                model_name=self.model, api_key=self.api_key,
                base_url=self.base_url, reasoning_effort=self.reasoning_effort,
            )
        elif self.backend == "gemini":
            from ihbench.clients.gemini_client import GeminiChatClient

            return GeminiChatClient(model_name=self.model, api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported judge backend: {self.backend}")
