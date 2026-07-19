#!/usr/bin/env python3
"""Single-checkpoint vLLM serving benchmark — decode TPS + load verification.

Run inside the vLLM environment (on this project: the WSL Ubuntu venv, since
vLLM and the quantization stack can't share one env — see README):

    python scripts/vllm_bench.py --model <path-or-hf-id> [--quantization compressed-tensors]

Emits one JSON line to stdout (machine-readable, appended to the leaderboard
by hand or script):

    {"model": ..., "quantization": ..., "loaded": true, "decode_tps": ...,
     "ttft_ms": ..., "output_tokens": ..., "error": null}

Design notes:
- temperature=0, fixed prompt, min/max tokens forced equal -> deterministic,
  stable single-stream decode TPS.
- VRAM is NOT reported here: vLLM preallocates gpu_memory_utilization worth of
  VRAM regardless of model size, so a peak-VRAM reading measures the config
  knob, not the model. The leaderboard's VRAM column comes from the HF runtime.
- A load failure is a result, not a crash: the JSON line carries the error so
  an unsupported (checkpoint, GPU) combination lands in the table as an honest
  "does not load" instead of vanishing.
"""
from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM single-stream decode benchmark")
    p.add_argument("--model", required=True, help="HF id or checkpoint path")
    p.add_argument("--quantization", default=None,
                   help="vLLM quantization flag (compressed-tensors, modelopt, ...)")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--cuda-graphs", action="store_true",
                   help="Enable CUDA graph capture / compilation (default off: engine "
                        "init with graphs spikes host RAM, which OOMs a constrained WSL "
                        "VM; eager slightly underreports TPS but runs everywhere)")
    args = p.parse_args()

    result = {
        "model": args.model,
        "quantization": args.quantization,
        "loaded": False,
        "decode_tps": None,
        "ttft_ms": None,
        "output_tokens": None,
        "error": None,
    }

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            enforce_eager=not args.cuda_graphs,
        )
        result["loaded"] = True

        prompt = "Explain the theory of relativity in simple terms, step by step."
        warmup = SamplingParams(temperature=0, max_tokens=32)
        llm.generate([prompt], warmup)

        # TTFT: single-token generation.
        t0 = time.perf_counter()
        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=1))
        result["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Decode TPS: forced-length single stream.
        sp = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                            min_tokens=args.output_tokens)
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        dt = time.perf_counter() - t0
        n = len(out[0].outputs[0].token_ids)
        result["output_tokens"] = n
        result["decode_tps"] = round(n / dt, 2)
    except Exception as exc:  # noqa: BLE001 - load/serve failure is a reportable result
        result["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"

    print("VLLM_BENCH_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
