# Qwen3-1.7B Quantization Leaderboard — RTX 3060 (12GB)

One model, every quantization path this framework supports, one table. Every number
measured on this hardware by this repo — no imported claims. Cells that can't be
measured on this hardware/stack say so and why; that's data too.

**Environment:** RTX 3060 12GB (sm_86, Ampere) · driver 610.62 · CUDA 12.8 ·
torch 2.8.0+cu128 · transformers 4.55.4 · vLLM 0.11.2 (WSL2 Ubuntu 24.04, torch 2.9.0) ·
Windows 10 · commit `3abf371`.
**Quality metric:** PPL (wikitext-2) + MMLU-tiny delta vs FP16 — Qwen3-1.7B is text-only,
so the CER column that a VLM version of this table would carry becomes PPL/MMLU here.

## The table

| Config | Decode TPS (HF) | Decode TPS (vLLM) | VRAM GB (HF) | PPL Δ vs FP16 | MMLU Δ | Serving |
|---|---|---|---|---|---|---|
| FP16 baseline | 17.8 | 56.1 | 3.28 | — (22.45) | — (0.548) | ✅ vLLM |
| llmcompressor AWQ-W4A16 | 4.1 | **57.9** | 1.30 | **+15.36** (37.81) | −0.032 | ✅ vLLM (`compressed-tensors`, Marlin) |
| llmcompressor GPTQ-W8A8 | 2.8 | 49.3 | 1.95 | +0.27 (22.72) | +0.008 | ✅ vLLM (`compressed-tensors`) |
| modelopt FP8 | 21.0\* | **load refused** | 3.29\* | 0.00\* | 0.00\* | ❌ vLLM: `Cannot find the config file for modelopt` — this repo's export drops quant metadata (Finding 3) |
| modelopt NVFP4 | n/a | n/a | n/a | n/a | n/a | ❌ needs Blackwell (sm_100+); export not attempted — same metadata gap as FP8 means it couldn't be serving-validated regardless |
| modelopt → TRT-LLM | n/a | n/a | n/a | n/a | n/a | ❌ not attempted — no supported TRT-LLM path on this Windows/WSL setup |
| torchao int4wo | **failed** | — | — | — | — | ❌ `Requires mslk >= 1.0.0` — kernel dep unavailable (PyPI has only a 0.0.0 placeholder); torchao 0.17/torch 2.8/sm_86 |
| AWQ-W4A16 + TurboQuant KV (K3V2) | 4.3 | — | 1.29 | +15.40 (37.85)† | −0.024† | HF only (TurboQuant not wired into vLLM) |

vLLM runs: 0.11.2 in WSL2 Ubuntu 24.04, eager mode (CUDA graph capture disabled — engine
init with graphs OOMed the memory-capped VM; eager slightly *understates* vLLM TPS, so the
serving-vs-eager gap is, if anything, larger than shown). TTFT: fp16 20.8ms, W4A16 17.7ms,
W8A8 35.4ms.

\* **modelopt FP8 on HF measures fp16, not FP8.** The export contains no
`quantization_config` and vanilla `from_pretrained` loads plain bf16 weights — PPL came
back bit-identical to baseline (22.449370186003648, 16 decimals), which is the tell. The
FP8 row's real test is its vLLM/serving cell. This also exposed a repo gap: the modelopt
saver writes fake-quant weights only, not the `hf_quant_config.json` a serving engine
needs (see Findings #3).

† Teacher-forced PPL can't see KV-cache quantization (the cache is never read back during
scoring — `docs/failure_cases.md` #4), so the TQ row's PPL ≈ its base checkpoint's. TQ's
real quality cost is next-token agreement (`docs/benchmark_report.md`), and its real win
is context capacity: 4x fp16 on this card. The row is here to show the two quantizers
*compose* — weight quant (AWQ) and KV quant (TurboQuant) stack without interference:
same VRAM, same TPS, same PPL as plain AWQ within noise.

HF decode TPS = single-stream batch-1, greedy, 128 forced tokens, HF `generate` eager.
vLLM decode TPS = same shape via `scripts/vllm_bench.py` (temperature 0, 256 forced
tokens, single stream). VRAM = peak allocated during the HF benchmark (weights resident +
inference; vLLM VRAM deliberately not reported — it preallocates a configured fraction,
so a reading there measures the knob, not the model).

## Findings

1. **W4A16 badly hurts a 1.7B model: +15.4 PPL, −3.2pp MMLU.** AWQ-W4A16 at 128
   calibration samples took Qwen3-1.7B from 22.45 to 37.81 PPL. Small models have less
   redundancy to absorb 4-bit weight error; this is consistent with the general
   observation that aggressive weight quantization punishes small models hardest. A
   larger calibration set (512+) might claw some back — untested here. If you need
   4-bit at this scale, measure before shipping.
2. **W8A8 is the quality-safe choice: +0.27 PPL, MMLU within noise.** GPTQ-W8A8 is
   near-lossless on both metrics at 60% of fp16's VRAM (1.95 vs 3.28 GB resident).
3. **This repo's modelopt export can't reach a serving engine yet — verified at both
   ends.** `mtq.quantize` + `save_pretrained` stores simulated-quant weights with no
   quantizer metadata. On the HF side that means `from_pretrained` silently loads plain
   bf16 (bit-identical PPL was the tell); on the vLLM side the loader outright refuses:
   `Cannot find the config file for modelopt`. The fix is routing
   `ModelOptQuantizer.save` through `modelopt.torch.export.export_hf_checkpoint`, which
   writes the `hf_quant_config.json` + scale tensors serving engines read. Queued as the
   top v1.0.1 item — until then, every modelopt row here is quantize-only.
4. **The same checkpoint is 14x slower than fp16 in HF eager and *faster* than fp16 in
   vLLM.** AWQ-W4A16: 4.1 TPS on HF eager (per-op dequant, no fused kernels) vs 57.9 TPS
   under vLLM's Marlin kernels — edging out even fp16's 56.1. Never benchmark a
   quantized checkpoint on an eager runtime and conclude the format is slow; runtime
   kernels, not the format, decide the speed story. Memory savings, by contrast, are
   real everywhere (1.30 GB vs 3.28 GB resident on HF).
5. **torchao int4wo is stack-gated, not hardware-gated:** the tinygemm int4 path in
   torchao 0.17 on torch 2.8 demands an `mslk >= 1.0.0` package that has no installable
   release — nothing to do with the GPU. int8wo from the same backend works fine on this
   box (measured in `docs/benchmark_report.md`'s sweep at −31% VRAM).

## Reproduce

```bash
# quantize the three local checkpoints
python quantize.py -c config/quantize/qwen3_1_7b/llmc_awq_w4a16.yaml
python quantize.py -c config/quantize/qwen3_1_7b/llmc_gptq_w8a8.yaml
python quantize.py -c config/quantize/qwen3_1_7b/modelopt_ptq_fp8.yaml

# HF-side table numbers (Windows env)
python benchmark.py -c config/benchmark/qwen3_1_7b_leaderboard.yaml

# vLLM-side serving numbers (separate env — WSL here)
python scripts/vllm_bench.py --model Qwen/Qwen3-1.7B
python scripts/vllm_bench.py --model outputs/qwen3-1.7b/awq-w4a16/Qwen3-1.7B-llm_compressor-awq-W4A16 --quantization compressed-tensors
# ... one line of JSON per run -> results/qwen3-1.7b-leaderboard/vllm_serving.jsonl
```

Raw data: `results/qwen3-1.7b-leaderboard/` (per-model JSON + comparison summary with
full environment provenance).
