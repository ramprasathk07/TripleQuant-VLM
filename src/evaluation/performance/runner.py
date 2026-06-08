"""OOM-safe orchestration of performance + context-capacity metrics.

Every measurement is run through :func:`oom_safe`, which catches CUDA
out-of-memory (and the RuntimeError variant some kernels raise), frees the
allocator cache, and returns a structured ``{"oom": True, ...}`` marker. This
guarantees that a single OOM — common when probing context length or sweeping
batch sizes on a small GPU — never aborts the surrounding benchmark loop.

Context-length capacity ("how many tokens fit on this GPU") is exposed via the
runtime's ``measure_max_context()``:
  - HFRuntime  → binary-searches the real forward-pass OOM boundary.
  - VLLMRuntime → reports the engine's ``max_model_len`` + KV-cache token budget.
"""
from __future__ import annotations

import gc
import logging
import traceback
from typing import Callable

logger = logging.getLogger(__name__)

# Imported lazily so this module stays importable without a CUDA torch build
# (e.g. during --dry-run config validation).
try:
    import torch
    _TORCH_OK = True
except Exception:  # pragma: no cover - torch always present in real runs
    _TORCH_OK = False

_DEFAULT_PERF_PROMPT = "Explain the theory of relativity in simple terms, step by step."



# GPU memory + OOM helpers


def clear_gpu_memory() -> None:
    """Release cached allocator blocks and run a GC pass. No-op without CUDA."""
    gc.collect()
    if _TORCH_OK and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def is_oom_error(exc: BaseException) -> bool:
    """True if *exc* is a CUDA out-of-memory condition.

    Covers both ``torch.cuda.OutOfMemoryError`` and the plain ``RuntimeError``
    with an "out of memory" message that some custom / quantized kernels raise.
    """
    if _TORCH_OK and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cublas_status_alloc_failed" in msg


def oom_safe(label: str, fn: Callable[[], dict | list]) -> dict | list:
    """Run ``fn`` and never let an OOM (or other error) escape.

    Returns ``fn``'s value on success. On OOM → ``{"oom": True, "detail": ...}``;
    on any other exception → ``{"error": "<traceback>"}``. The GPU cache is freed
    afterwards either way so the next metric starts from a clean allocator.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - intentional catch-all for crash safety
        if is_oom_error(exc):
            logger.warning("[perf] '%s' hit CUDA OOM — skipping (benchmark continues).", label)
            clear_gpu_memory()
            return {"oom": True, "detail": str(exc)[:300]}
        logger.warning("[perf] '%s' failed.", label, exc_info=True)
        clear_gpu_memory()
        return {"error": traceback.format_exc()}



# Orchestration
def run_perf_metrics(runtime, config, perf_prompt: str = _DEFAULT_PERF_PROMPT,
                     tq_enabled: bool = False) -> dict:
    """Run all enabled perf metrics for *runtime*, each guarded against OOM.

    Args:
        runtime:     A loaded runtime (HFRuntime / VLLMRuntime).
        config:      BenchmarkConfig — reads ``metrics.perf`` and ``latency``.
        perf_prompt: Prompt used for latency/throughput trials.
        tq_enabled:  Whether this entry has TurboQuant on (gates the TQ-only
                     bits/accuracy sweep so it isn't duplicated on the baseline).

    Returns:
        dict keyed by metric. Any metric that OOMs maps to ``{"oom": True}``;
        any that errors maps to ``{"error": ...}`` — never raises.
    """
    out: dict = {}
    wanted = config.metrics.perf
    lat = config.latency

    if "ttft" in wanted or "tpot" in wanted:
        out["latency"] = oom_safe(
            "ttft_tpot",
            lambda: runtime.measure_ttft_tpot(perf_prompt, n=lat.num_requests),
        )

    if "throughput" in wanted:
        out["throughput"] = oom_safe(
            "throughput",
            lambda: runtime.measure_throughput(
                perf_prompt,
                batch_sizes=lat.batch_sizes,
                output_len=lat.output_lens[0],
            ),
        )

    if "max_context" in wanted:
        if hasattr(runtime, "measure_max_context"):
            out["max_context"] = oom_safe("max_context", runtime.measure_max_context)
        else:
            out["max_context"] = {"skipped": "runtime has no measure_max_context()"}

    if "ctx_sweep" in wanted:
        if hasattr(runtime, "measure_ctx_sweep"):
            out["ctx_sweep"] = oom_safe(
                "ctx_sweep",
                lambda: runtime.measure_ctx_sweep(lat.ctx_sweep),
            )
        else:
            out["ctx_sweep"] = {"skipped": "runtime has no measure_ctx_sweep()"}

    # TQ bits/accuracy sweep characterizes TurboQuant — only run it on the TQ entry
    # so the baseline doesn't duplicate the identical (model-only) result.
    if "tq_bits_sweep" in wanted and tq_enabled:
        if hasattr(runtime, "measure_bits_accuracy"):
            out["tq_bits_sweep"] = oom_safe(
                "tq_bits_sweep",
                lambda: runtime.measure_bits_accuracy(
                    lat.tq_bits_pairs, lat.tq_sweep_lengths, lat.tq_sweep_ring),
            )
        else:
            out["tq_bits_sweep"] = {"skipped": "runtime has no measure_bits_accuracy()"}

    return out
