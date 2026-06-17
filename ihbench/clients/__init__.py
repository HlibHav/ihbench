"""Model clients for IHBench inference.

Each `*_client.py` file in this package self-registers via the
`@register_client(...)` decorator (see `registry.py`). On import of this
package we auto-discover all client files so that `get_client(model_id)`
returns the right class without a central dispatch table.

Imports inside client files (transformers, librosa, model-specific classes)
must be lazy — done inside `_load_model` or `_build_inputs` — so that
missing optional dependencies for one model don't break the package import.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from ihbench.clients.base import APIClient, GenerationConfig, GenericMessage, LocalClient, ModelClient
from ihbench.clients.registry import get_client, list_registered, register_client

# Auto-discover client files. The HF base client is excluded because it is the
# fallback used by `get_client` and importing it here would just slow startup.
for _p in Path(__file__).parent.glob("*_client.py"):
    if _p.stem in {"hf_client"}:
        continue
    importlib.import_module(f"ihbench.clients.{_p.stem}")

__all__ = [
    "ModelClient",
    "APIClient",
    "LocalClient",
    "GenerationConfig",
    "GenericMessage",
    "register_client",
    "get_client",
    "list_registered",
]
