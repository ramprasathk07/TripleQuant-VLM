# src/runtimes/vllm_runtime.py
"""
VLLMRuntime – vLLM backend.

Supports:
  - Text-only LLM inference via vLLM's LLM engine
  - Async-capable SamplingParams-based generation
  - Throughput and latency (TTFT / TPOT) profiling
  - Quantization via vLLM's native AWQ / GPTQ / FP8 support
  - Multi-GPU tensor parallelism
  - Explicit engine teardown with full GPU memory release

Limitations (by design):
  - forward_logits()        → raises NotImplementedError (vLLM hides logits)
  - generate_with_logits()  → raises NotImplementedError
  - generate_vlm()          → raises NotImplementedError (text-only runtime)
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..base import RuntimeBase

logger = logging.getLogger(__name__)

# ── vLLM import (optional at module level, fail loudly only when used) ────────
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Internal helpers
def _require_vllm() -> None:
    if not VLLM_AVAILABLE:
        raise ImportError(
            "VLLMRuntime requires the `vllm` package. "
            "Install it with: pip install vllm"
        )

def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile from a sorted list of values.

    Args:
        sorted_data: Pre-sorted list of floats.
        p: Percentile rank (0.0-1.0).

    Returns:
        Value at the requested percentile.
    """
    idx = max(0, min(int(len(sorted_data) * p), len(sorted_data) - 1))
    return sorted_data[idx]

def _timing_stats(times_ms: list[float]) -> dict:
    """Compute timing statistics (mean, percentiles, min/max).

    Args:
        times_ms: List of timing measurements in milliseconds.

    Returns:
        Dict with keys: mean, p50, p95, p99, min, max.
    """
    s = sorted(times_ms)
    return {
        "mean": statistics.mean(s),
        "p50":  _percentile(s, 0.50),
        "p95":  _percentile(s, 0.95),
        "p99":  _percentile(s, 0.99),
        "min":  s[0],
        "max":  s[-1],
    }

def _resolve_vllm_dtype(dtype_str: Optional[str]) -> str:
    """Map dtype string to vLLM-compatible dtype string.

    Args:
        dtype_str: Dtype name (aliases: 'fp16'/'float16', 'bf16'/'bfloat16',
                   'fp32'/'float32', 'auto' or None).

    Returns:
        vLLM-compatible dtype string ('float16', 'bfloat16', 'float32', 'auto').

    Raises:
        ValueError: If dtype_str is not recognized.
    """
    _map = {
        "fp16":     "float16",
        "float16":  "float16",
        "bf16":     "bfloat16",
        "bfloat16": "bfloat16",
        "fp32":     "float32",
        "float32":  "float32",
        None:       "auto",
        "auto":     "auto",
    }
    key = dtype_str.lower() if dtype_str else None
    if key not in _map:
        raise ValueError(
            f"Unknown dtype '{dtype_str}' for vLLM. "
            f"Choose from: {[k for k in _map if k is not None]}"
        )
    return _map[key]

def _resolve_vllm_quantization(quant: Optional[str]) -> Optional[str]:
    """Map quantization string to vLLM-accepted quantization type.

    Args:
        quant: Quantization method name (e.g., 'awq', 'gptq', 'fp8', 'compressed-tensors').

    Returns:
        vLLM quantization string, or None for no quantization.

    Raises:
        ValueError: If quantization method is not supported by vLLM (e.g., BitsAndBytes).

    Note:
        vLLM does NOT support BitsAndBytes (int4, int8, nf4). Use pre-quantized
        model variants (AWQ, GPTQ, llm-compressor, ModelOpt) instead.
    """
    if quant is None or quant.lower() in ("none", ""):
        return None

    q = quant.lower()
    _supported = {
        "awq", "gptq", "squeezellm", "fp8",
        "compressed-tensors", "modelopt", "modelopt_fp4",
    }
    _unsupported = {"int4", "int8", "nf4", "bnb"}

    if q in _unsupported:
        raise ValueError(
            f"vLLM does not support BitsAndBytes quantization ('{quant}'). "
            "Use AWQ / GPTQ / compressed-tensors / ModelOpt model variants instead."
        )
    if q not in _supported:
        raise ValueError(
            f"Unknown vLLM quantization '{quant}'. "
            f"Supported: {_supported}"
        )
    return q



# VLLMRuntime
class VLLMRuntime(RuntimeBase):
    """vLLM backend for high-throughput text-only LLM inference.

    Wraps vLLM's LLM engine for fast batch/streaming generation. Supports
    native AWQ/GPTQ quantization, tensor parallelism, and GPU memory tuning.
    Text-only: VLM, logit-returning, and quantization-free inference methods
    raise NotImplementedError.

    Attributes:
        name: Identifier 'vllm' for runtime registry.

    Limitations:
        - forward_logits() raises NotImplementedError (vLLM hides logits).
        - generate_with_logits() raises NotImplementedError.
        - generate_vlm() raises NotImplementedError (text-only).
        - No BitsAndBytes support; use pre-quantized models (AWQ, GPTQ).

    Usage:
        runtime = VLLMRuntime()
        runtime.load(entry)
        outputs = runtime.generate(["Hello!"], max_new_tokens=64)
        runtime.unload()
    """

    name = "vllm"

    def __init__(self) -> None:
        """Initialize VLLMRuntime, checking vllm availability."""
        _require_vllm()

        self._llm:      Optional["LLM"] = None
        self._model_id: str             = ""
        self._tokenizer                 = None

    def load(self, entry: "BenchmarkModelEntry") -> None:
        """Initialize vLLM engine from BenchmarkModelEntry config.

        Creates and configures the vLLM LLM engine with model, quantization,
        dtype, and tensor parallelism settings.

        Args:
            entry: BenchmarkModelEntry specifying model path, quantization,
                   dtype, tensor_parallel_size, gpu_memory_utilization, max_model_len.

        Raises:
            RuntimeError: If engine is already loaded.
            ValueError: If BitsAndBytes quantization is requested (not supported by vLLM).
        """
        if self._llm is not None:
            raise RuntimeError(
                "VLLMRuntime: engine already loaded. Call unload() first."
            )

        model_id  = entry.path
        self._model_id = model_id

        quant     = _resolve_vllm_quantization(getattr(entry, "vllm_quantization",      None))
        dtype     = _resolve_vllm_dtype(        getattr(entry, "dtype",                 None))
        tp_size   =                              getattr(entry, "tensor_parallel_size",  1)
        gpu_mem   =                              getattr(entry, "gpu_memory_utilization", 0.85)
        max_len   =                              getattr(entry, "max_model_len",         None)
        trust_rc  =                              getattr(entry, "trust_remote_code",     False)

        logger.info(
            f"[VLLMRuntime] Loading '{model_id}' | "
            f"quant={quant} | dtype={dtype} | "
            f"tp={tp_size} | gpu_mem={gpu_mem}"
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Build kwargs conditionally (max_model_len is optional)
        engine_kwargs = dict(
            model                  = model_id,
            dtype                  = dtype,
            quantization           = quant,
            tensor_parallel_size   = tp_size,
            gpu_memory_utilization = gpu_mem,
            trust_remote_code      = trust_rc,
        )
        if max_len is not None:
            engine_kwargs["max_model_len"] = max_len

        self._llm       = LLM(**engine_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

        logger.info(
            f"[VLLMRuntime] '{model_id}' ready. "
            f"Peak VRAM: {self.peak_vram_mb():.1f} MB"
        )

    def unload(self) -> None:
        """
        Tears down the vLLM engine and releases all GPU memory.

        Calls destroy_model_parallel() to cleanly shut down tensor-parallel
        workers before deleting the engine object.
        Safe to call multiple times.
        """
        logger.info(f"[VLLMRuntime] Unloading '{self._model_id}' …")

        if self._llm is not None:
            try:
                destroy_model_parallel()
                logger.debug("[VLLMRuntime] Model parallel group destroyed.")
            except Exception as e:
                logger.warning(f"[VLLMRuntime] destroy_model_parallel() failed: {e}")

            del self._llm
            self._llm = None

        self._tokenizer = None
        self._model_id  = ""

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("[VLLMRuntime] Unload complete.")

    
    # Text generation
    def generate(
        self,
        prompts:        list[str],
        max_new_tokens: int,
        temperature:    float = 0.0,
    ) -> list[str]:
        """
        Batched text generation via vLLM.

        Args:
            prompts:        Input strings (batch).
            max_new_tokens: Token budget per prompt.
            temperature:    0.0 → greedy; > 0.0 → sampling.

        Returns:
            List of decoded strings, one per prompt (prompt not included).
        """
        self._check_loaded()

        sampling_params = SamplingParams(
            max_tokens  = max_new_tokens,
            temperature = temperature if temperature > 0.0 else 0.0,
            # vLLM uses temperature=0 for greedy internally
        )

        request_outputs = self._llm.generate(prompts, sampling_params)

        # vLLM returns outputs in submission order
        return [
            output.outputs[0].text
            for output in request_outputs
        ]

    def generate_with_logits(
        self,
        prompts:        list[str],
        max_new_tokens: int,
    ) -> Tuple[list[str], list[Tensor]]:
        """
        NOT SUPPORTED by vLLM — logit tensors are not exposed by the engine.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "generate_with_logits() is not supported by VLLMRuntime. "
            "Use HFRuntime for logit-level access."
        )

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """
        NOT SUPPORTED by vLLM — the engine does not expose raw logits.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "forward_logits() is not supported by VLLMRuntime. "
            "Switch to HFRuntime for PPL / KL-divergence evaluation."
        )

    
    # VLM generation
    def generate_vlm(
        self,
        image,
        prompt:         str,
        max_new_tokens: int,
    ) -> str:
        """
        NOT SUPPORTED — VLLMRuntime is text-only.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "generate_vlm() is not supported by VLLMRuntime. "
            "Use HFRuntime with entry.is_vlm=True for VLM inference."
        )

    
    # Latency profiling
    def measure_ttft_tpot(self, prompt: str, n: int = 100) -> dict:
        """
        Measures Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT)
        using vLLM's request-level timing.

        Strategy:
            - TTFT: generate exactly 1 token and time the full request.
              vLLM schedules the prefill + one decode step.
            - TPOT: generate OUTPUT_TOKENS tokens, subtract TTFT,
              divide by (OUTPUT_TOKENS - 1).

        Args:
            prompt: Input string used for all trials.
            n:      Number of timed repetitions (first 5 discarded as warm-up).

        Returns:
            {
                "ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
                "ttft_ms_min",  "ttft_ms_max",
                "tpot_ms_mean", "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
                "tpot_ms_min",  "tpot_ms_max",
                "n_trials":     int,
            }
        """
        self._check_loaded()

        OUTPUT_TOKENS = 64
        WARMUP        = min(5, n)

        params_ttft = SamplingParams(max_tokens=1,            temperature=0.0)
        params_tpot = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0.0)

        ttft_times: list[float] = []
        tpot_times: list[float] = []

        for trial in range(n + WARMUP):
            # ── TTFT ──────────────────────────────────────────────────────────
            _sync_cuda()
            t0 = time.perf_counter()
            self._llm.generate([prompt], params_ttft)
            _sync_cuda()
            ttft_ms = (time.perf_counter() - t0) * 1_000

            # ── TPOT ──────────────────────────────────────────────────────────
            _sync_cuda()
            t1 = time.perf_counter()
            out = self._llm.generate([prompt], params_tpot)
            _sync_cuda()
            total_ms    = (time.perf_counter() - t1) * 1_000
            n_generated = len(out[0].outputs[0].token_ids)

            tpot_ms = (
                (total_ms - ttft_ms) / (n_generated - 1)
                if n_generated > 1
                else total_ms
            )

            if trial >= WARMUP:
                ttft_times.append(ttft_ms)
                tpot_times.append(max(tpot_ms, 0.0))

        ttft_stats = _timing_stats(ttft_times)
        tpot_stats = _timing_stats(tpot_times)

        return {
            "ttft_ms_mean": ttft_stats["mean"],
            "ttft_ms_p50":  ttft_stats["p50"],
            "ttft_ms_p95":  ttft_stats["p95"],
            "ttft_ms_p99":  ttft_stats["p99"],
            "ttft_ms_min":  ttft_stats["min"],
            "ttft_ms_max":  ttft_stats["max"],
            "tpot_ms_mean": tpot_stats["mean"],
            "tpot_ms_p50":  tpot_stats["p50"],
            "tpot_ms_p95":  tpot_stats["p95"],
            "tpot_ms_p99":  tpot_stats["p99"],
            "tpot_ms_min":  tpot_stats["min"],
            "tpot_ms_max":  tpot_stats["max"],
            "n_trials":     n,
        }

    
    # Throughput profiling
    

    def measure_throughput(
        self,
        prompt:      str,
        batch_sizes: list[int],
        output_len:  int,
    ) -> list[dict]:
        """
        Sweeps batch sizes and records tokens/sec, latency, and OOM status.

        vLLM handles batching internally — we replicate the prompt bs times
        and submit all requests in a single generate() call.

        Each batch size is timed over 3 runs; median latency is reported.

        Args:
            prompt:      Single prompt string replicated across the batch.
            batch_sizes: List of batch sizes, e.g. [1, 4, 8, 16, 32].
            output_len:  Fixed number of output tokens per request.

        Returns:
            [
                {
                    "batch_size":     int,
                    "tokens_per_sec": float,
                    "latency_ms":     float,
                    "total_tokens":   int,
                    "oom":            bool,
                },
                ...
            ]
        """
        self._check_loaded()

        RUNS_PER_BATCH  = 3
        sampling_params = SamplingParams(
            max_tokens  = output_len,
            temperature = 0.0,
        )
        results = []

        for bs in batch_sizes:
            prompts      = [prompt] * bs
            latencies_ms = []
            oom          = False

            for run in range(RUNS_PER_BATCH):
                try:
                    _sync_cuda()
                    t0 = time.perf_counter()
                    self._llm.generate(prompts, sampling_params)
                    _sync_cuda()
                    latencies_ms.append((time.perf_counter() - t0) * 1_000)

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"[VLLMRuntime] OOM at batch_size={bs}. "
                        "Skipping remaining runs for this batch size."
                    )
                    oom = True
                    torch.cuda.empty_cache()
                    break

                except Exception as e:
                    logger.error(
                        f"[VLLMRuntime] Unexpected error at batch_size={bs}, "
                        f"run={run}: {e}"
                    )
                    oom = True
                    break

            if oom or not latencies_ms:
                results.append({
                    "batch_size":     bs,
                    "tokens_per_sec": 0.0,
                    "latency_ms":     float("inf"),
                    "total_tokens":   0,
                    "oom":            True,
                })
                continue

            median_latency_ms = sorted(latencies_ms)[RUNS_PER_BATCH // 2]
            total_tokens      = bs * output_len
            tokens_per_sec    = total_tokens / (median_latency_ms / 1_000)

            results.append({
                "batch_size":     bs,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "latency_ms":     round(median_latency_ms, 2),
                "total_tokens":   total_tokens,
                "oom":            False,
            })

            logger.info(
                f"[VLLMRuntime] bs={bs:>3} | "
                f"tps={tokens_per_sec:>8.1f} | "
                f"latency={median_latency_ms:>7.1f} ms"
            )

        return results

    
    # Memory
    

    def peak_vram_mb(self) -> float:
        """
        Returns peak VRAM allocated (in MB) since last reset.
        Returns 0.0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Max context  (read from engine config — vLLM pre-sizes the KV cache, no probe)
    

    def measure_max_context(self) -> dict:
        """Report the engine's max context + KV-cache capacity.

        Unlike the HF probe (which binary-searches for the OOM boundary), vLLM
        fixes ``max_model_len`` at load and pre-allocates the KV cache, so the
        usable context is known exactly from the engine config. Returns the same
        ``model_max_positions``/``measured_max_tokens`` keys as HFRuntime plus
        vLLM-specific KV stats when available.
        """
        self._check_loaded()
        out: dict = {
            "model_max_positions": None,
            "measured_max_tokens": None,
            "oom_at_tokens":       None,
            "capped_by":           "vllm_max_model_len",
        }
        try:
            mcfg = self._llm.llm_engine.model_config
            max_len = int(mcfg.max_model_len)
            out["model_max_positions"] = max_len
            out["measured_max_tokens"] = max_len   # guaranteed loadable by construction
        except Exception as e:
            logger.warning("[VLLMRuntime] could not read max_model_len: %s", e)

        # KV-cache block stats (attribute path varies across vLLM versions).
        try:
            cache_cfg = self._llm.llm_engine.cache_config
            num_blocks = getattr(cache_cfg, "num_gpu_blocks", None)
            block_size = getattr(cache_cfg, "block_size", None)
            if num_blocks and block_size:
                out["kv_cache_tokens"] = int(num_blocks) * int(block_size)
                out["kv_gpu_blocks"]   = int(num_blocks)
                out["kv_block_size"]   = int(block_size)
        except Exception:
            pass
        return out

    
    # Tokenizer access (convenience)
    

    @property
    def tokenizer(self):
        """
        Exposes the underlying HF tokenizer from the vLLM engine.
        Useful for token counting, prompt formatting, and MMLU scoring.

        Raises:
            RuntimeError: If called before load().
        """
        self._check_loaded()
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string to token IDs using the vLLM tokenizer.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        self._check_loaded()
        return self._tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes token IDs back to a string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        self._check_loaded()
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def token_count(self, text: str) -> int:
        """
        Returns the number of tokens in a string.
        Useful for prompt length checks before submission.

        Args:
            text: Input string.

        Returns:
            Integer token count.
        """
        return len(self.encode(text))

    
    # score_choices  (MMLU scoring — log-prob approximation via generation)
    

    def score_choices(
        self,
        prompt:  str,
        choices: list[str],
    ) -> dict[str, float]:
        """
        Approximates log-probabilities for single-token choices via vLLM's
        logprobs sampling parameter.

        vLLM can return top-k log-probs for each generated token when
        logprobs=k is set in SamplingParams. We generate 1 token and
        read the log-prob of each choice token from the returned distribution.

        Args:
            prompt:  Full prompt string ending with e.g. "Answer:".
            choices: Single-token label strings, e.g. ["A", "B", "C", "D"].

        Returns:
            {"A": -0.32, "B": -1.45, "C": -2.10, "D": -3.88}
            Missing tokens (not in top-k) receive -inf.

        Note:
            vLLM's logprobs only covers the top-k vocabulary. If a choice
            token falls outside the top-k window it will be -inf.
            For MMLU A/B/C/D this is extremely unlikely with logprobs=20.
        """
        self._check_loaded()

        LOGPROBS_K = 20   # retrieve top-20 log-probs at the first decode step

        sampling_params = SamplingParams(
            max_tokens = 1,
            temperature= 0.0,
            logprobs   = LOGPROBS_K,
        )

        outputs = self._llm.generate([prompt], sampling_params)
        # outputs[0].outputs[0].logprobs is a list of dicts (one per step)
        # Each dict: {token_id: Logprob(logprob=float, decoded_token=str)}
        step_logprobs = outputs[0].outputs[0].logprobs[0]   # first (only) step

        # Build token_id → log_prob lookup
        id_to_lp: dict[int, float] = {
            tid: lp_obj.logprob
            for tid, lp_obj in step_logprobs.items()
        }

        scores: dict[str, float] = {}
        for label in choices:
            # Encode " A" (with leading space) as most tokenizers do
            token_ids = self._tokenizer.encode(
                f" {label}", add_special_tokens=False
            )
            if not token_ids:
                scores[label] = float("-inf")
                continue

            tid = token_ids[0]
            scores[label] = id_to_lp.get(tid, float("-inf"))

        return scores

    
    # Batch generate with per-request metadata  (convenience)
    

    def generate_with_metadata(
        self,
        prompts:        list[str],
        max_new_tokens: int,
        temperature:    float = 0.0,
    ) -> list[dict]:
        """
        Extended generation that returns output text plus per-request metadata.

        Returns:
            List of dicts, one per prompt:
            [
                {
                    "text":           str,    – generated text
                    "token_ids":      list[int],
                    "finish_reason":  str,    – "stop" | "length"
                    "prompt_tokens":  int,
                    "output_tokens":  int,
                },
                ...
            ]
        """
        self._check_loaded()

        sampling_params = SamplingParams(
            max_tokens  = max_new_tokens,
            temperature = temperature if temperature > 0.0 else 0.0,
        )

        request_outputs = self._llm.generate(prompts, sampling_params)

        results = []
        for req_out in request_outputs:
            completion    = req_out.outputs[0]
            prompt_tokens = len(req_out.prompt_token_ids)
            results.append({
                "text":          completion.text,
                "token_ids":     list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "prompt_tokens": prompt_tokens,
                "output_tokens": len(completion.token_ids),
            })

        return results

    
    # Internal guards
    

    def _check_loaded(self) -> None:
        """Raises RuntimeError if load() has not been called yet."""
        if self._llm is None:
            raise RuntimeError(
                "VLLMRuntime: engine is not loaded. Call load(entry) first."
            )

    
    # Dunder helpers
    

    def __repr__(self) -> str:
        status = f"loaded='{self._model_id}'" if self._llm else "unloaded"
        return f"VLLMRuntime({status})"

    def __enter__(self) -> "VLLMRuntime":
        """Supports usage as a context manager."""
        return self

    def __exit__(self, *_) -> None:
        """Automatically unloads engine on context manager exit."""
        self.unload()