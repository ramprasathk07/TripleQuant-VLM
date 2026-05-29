"""Performance + context-capacity metrics (OOM-safe).

Groups all latency / throughput / context-length-capacity benchmarking behind a
single OOM-safe entry point so a GPU out-of-memory at any stage degrades to a
recorded ``{"oom": ...}`` marker instead of aborting the whole benchmark run.
"""
from .runner import (
    clear_gpu_memory,
    is_oom_error,
    oom_safe,
    run_perf_metrics,
)

__all__ = [
    "clear_gpu_memory",
    "is_oom_error",
    "oom_safe",
    "run_perf_metrics",
]
