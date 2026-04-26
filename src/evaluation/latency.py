"""
Latency profiling using vLLM.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class VLLMLatencyProfiler:
    """Measures TTFT, TPOT, and throughput via vLLM.

    Args:
        model_path: HuggingFace model ID or path to locally saved model.
        gpu_memory_utilization: Fraction of GPU memory to reserve for vLLM.
        max_model_len: Maximum sequence length for the vLLM engine.
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
    ) -> None:
        logger.info(
            "Initialising vLLM engine for %s (mem_util=%.2f)",
            model_path,
            gpu_memory_utilization,
        )
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def measure_latency(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        warmup_runs: int = 2,
        timed_runs: int = 5,
    ) -> dict:
        """Benchmark TTFT, TPOT, and throughput.

        Args:
            prompt: Text prompt to generate from.
            max_new_tokens: Target generation length for throughput measurement.
            warmup_runs: Number of un-timed warm-up iterations.
            timed_runs: Number of timed iterations to average over.

        Returns:
            Dictionary with keys:
                ``avg_ttft_ms``, ``avg_throughput_tps``, ``avg_tpot_ms``,
                ``p50_ttft_ms``, ``p95_ttft_ms``, ``num_tokens``, ``iterations``.
        """
        ttft_params = SamplingParams(max_tokens=1, temperature=0)
        gen_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

        logger.info("Warm-up: %d runs …", warmup_runs)
        for _ in range(warmup_runs):
            self.llm.generate([prompt], ttft_params, use_tqdm=False)

        ttfts: list[float] = []
        throughputs: list[float] = []
        tpots: list[float] = []

        logger.info("Benchmarking latency over %d timed runs …", timed_runs)
        for i in range(timed_runs):
            # TTFT
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.llm.generate([prompt], ttft_params, use_tqdm=False)
            torch.cuda.synchronize()
            ttfts.append(time.perf_counter() - t0)

            # Full generation throughput
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            outputs = self.llm.generate([prompt], gen_params, use_tqdm=False)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            gen_time = t2 - t1
            num_tokens = len(outputs[0].outputs[0].token_ids)
            if num_tokens > 1:
                tps = num_tokens / gen_time
                throughputs.append(tps)
                tpots.append(1.0 / tps * 1000)  # ms

        ttfts_ms = [x * 1000 for x in ttfts]
        return {
            "avg_ttft_ms": float(np.mean(ttfts_ms)),
            "p50_ttft_ms": float(np.percentile(ttfts_ms, 50)),
            "p95_ttft_ms": float(np.percentile(ttfts_ms, 95)),
            "avg_throughput_tps": float(np.mean(throughputs)) if throughputs else 0.0,
            "avg_tpot_ms": float(np.mean(tpots)) if tpots else 0.0,
            "num_tokens": max_new_tokens,
            "iterations": timed_runs,
        }
