"""
GPU memory profiler.
"""
from __future__ import annotations

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def get_vram_usage() -> tuple[float, float]:
    """Return (current_vram_gb, peak_vram_gb).

    Returns ``(0.0, 0.0)`` when CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    current = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return current, peak


def get_model_size_gb(model) -> float:
    """Estimate in-memory model size in GB from parameter dtypes."""
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 3)


class MemoryProfiler:
    """Context-manager style profiler for GPU memory usage.

    Example::

        profiler = MemoryProfiler()
        result = profiler.profile(model.generate, **inputs, max_new_tokens=128)
        print(result)  # {"peak_vram_gb": ..., "delta_vram_gb": ...}
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def profile(self, func: Callable, *args, **kwargs) -> dict:
        """Call *func* with *args*/*kwargs* and return memory delta metrics.

        Returns:
            Dictionary with ``start_vram_gb``, ``end_vram_gb``,
            ``peak_vram_gb``, and ``delta_vram_gb``.
        """
        self.reset()
        start_mem, _ = get_vram_usage()

        result = func(*args, **kwargs)

        end_mem, peak_mem = get_vram_usage()
        metrics = {
            "start_vram_gb": round(start_mem, 3),
            "end_vram_gb": round(end_mem, 3),
            "peak_vram_gb": round(peak_mem, 3),
            "delta_vram_gb": round(peak_mem - start_mem, 3),
        }
        logger.debug("Memory profile: %s", metrics)
        return metrics
