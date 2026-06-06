# src/runtimes/factory.py
"""
Runtime factory — builds and loads the correct RuntimeBase subclass
from a string name and a BenchmarkModelEntry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base      import RuntimeBase
from .hf.hf_runtime   import HFRuntime
from .vllm.vllm_runtime import VLLMRuntime

if TYPE_CHECKING:
    from src.config.schemas import BenchmarkModelEntry

logger = logging.getLogger(__name__)

# Registry: add new runtimes here only
_REGISTRY: dict[str, type[RuntimeBase]] = {
    "hf":   HFRuntime,
    "vllm": VLLMRuntime,
}


def build_runtime(
    name:  str,
    entry: "BenchmarkModelEntry",
) -> RuntimeBase:
    """
    Instantiates and loads the requested runtime.

    Args:
        name:  Runtime name string — "hf" or "vllm".
        entry: BenchmarkModelEntry carrying model_id and runtime_cfg.

    Returns:
        A fully loaded RuntimeBase instance, ready for inference.

    Raises:
        ValueError:     If name is not a registered runtime.
        ImportError:    If the required backend package is not installed.
        RuntimeError:   If load() fails for any other reason.

    Example:
        >>> rt = build_runtime("hf", entry)
        >>> outputs = rt.generate(["Hello"], max_new_tokens=32)
        >>> rt.unload()
    """
    key = name.strip().lower()

    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown runtime '{name}'. "
            f"Registered runtimes: {list(_REGISTRY.keys())}"
        )

    runtime_cls = _REGISTRY[key]
    logger.info(f"[factory] Building runtime '{key}' for model '{entry.path}'")

    runtime = runtime_cls()
    runtime.load(entry)

    return runtime


def list_runtimes() -> list[str]:
    """Returns the names of all registered runtimes."""
    return list(_REGISTRY.keys())