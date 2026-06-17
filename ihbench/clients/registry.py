"""Client registry.

Each model-specific client decorates itself with `@register_client(<pattern>)` so
that `run_inference_hf.py` can look up the right class from an HF model ID
without having to maintain a central dispatch table. Patterns are `fnmatch`
globs against the HF model ID (e.g. "Qwen/Qwen3-Omni-*"). Adding a new model
becomes a single-file change.

Why fnmatch and not regex: model IDs only vary on size/variant suffixes, so
glob coverage is enough and the patterns stay readable in the decorator line.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Callable, Type

from ihbench.clients.base import LocalClient


_REGISTRY: list[tuple[str, Type[LocalClient]]] = []


def register_client(pattern: str) -> Callable[[Type[LocalClient]], Type[LocalClient]]:
    """Class decorator that registers a client for one or more HF model IDs.

    The first registered pattern that matches wins, so more-specific patterns
    should be registered before more-general ones. In practice each client
    file registers exactly one pattern, so ordering only matters across
    families.
    """

    def _decorate(cls: Type[LocalClient]) -> Type[LocalClient]:
        _REGISTRY.append((pattern, cls))
        return cls

    return _decorate


def get_client(model_name_or_path: str, **kwargs: Any) -> LocalClient:
    """Instantiate the registered client whose pattern matches `model_name_or_path`.

    Falls back to `HFLocalClient` for unmatched model IDs so adding a plain
    text-only HF model needs no client file at all.
    """
    for pattern, cls in _REGISTRY:
        if fnmatch.fnmatchcase(model_name_or_path, pattern):
            return cls(model_name_or_path=model_name_or_path, **kwargs)

    # Fallback for text-only / generic HF models
    from ihbench.clients.hf_client import HFLocalClient

    return HFLocalClient(model_name_or_path=model_name_or_path, **kwargs)


def list_registered() -> list[tuple[str, str]]:
    """Return [(pattern, class_name), ...] for debugging / `--list-clients`."""
    return [(pat, cls.__name__) for pat, cls in _REGISTRY]
