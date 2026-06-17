"""Abstract base classes for model clients."""

from __future__ import annotations

import abc
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_B64_PREFIX = "b64://"


def is_audio_content(content: Any) -> bool:
    """Return True iff `content` is either a base64 URI or an existing file path."""
    if not isinstance(content, str):
        return False
    if content.startswith(_B64_PREFIX):
        return True
    try:
        return Path(content).exists()
    except (OSError, ValueError):
        return False


def load_audio_b64(content: str) -> str:
    """Return base64-encoded WAV bytes from either a file path or a b64:// URI.

    Lets clients handle both IHBench-style (file path on disk) and
    AudioMultiChallenge-style (in-memory WAV passed as b64://...) audio
    inputs without special-casing the source.
    """
    if content.startswith(_B64_PREFIX):
        return content[len(_B64_PREFIX):]
    with open(content, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_audio_bytes(content: str) -> bytes:
    """Return raw WAV bytes from either a file path or a b64:// URI."""
    if content.startswith(_B64_PREFIX):
        return base64.b64decode(content[len(_B64_PREFIX):])
    return Path(content).read_bytes()


@dataclass
class GenericMessage:
    """A message in the wire format sent to models."""

    role: str  # "system", "user", "assistant"
    type: str  # "text" or "audio"
    content: str | np.ndarray  # str for text, float32 array at 16kHz for audio


@dataclass
class GenerationConfig:
    """Common generation parameters."""

    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 2048

    def to_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p, "max_new_tokens": self.max_new_tokens}


class ModelClient(abc.ABC):
    """Base interface for all model clients.

    All clients implement ``generate``, which takes a system message and a list
    of conversation messages and returns the model's text response.
    """

    @abc.abstractmethod
    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a single assistant response given conversation context.

        Args:
            system_message: The system prompt for the assistant.
            messages: Conversation history as a list of GenericMessage.
                      May contain both text and audio messages.
            config: Generation parameters (uses defaults if None).

        Returns:
            The generated text response.
        """
        ...


class APIClient(ModelClient):
    """Base class for API-backed model clients (OpenAI, Gemini, etc.)."""

    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name


class LocalClient(ModelClient):
    """Base class for locally-run models (HuggingFace + Accelerate, vLLM)."""

    def __init__(self, model_name_or_path: str, **kwargs: Any):
        self.model_name_or_path = model_name_or_path
