# Qwen3-1.7B Quantization Leaderboard — RTX 3060 (12GB)

One model, every quantization path this framework supports, one table. Every number
measured on this hardware by this repo — no imported claims. Cells that can't be
measured on this hardware/stack say so and why; that's data too.

**Environment:** RTX 3060 12GB (sm_86, Ampere) · driver 610.62 · CUDA 12.8 ·
torch 2.8.0+cu128 · transformers 4.55.4 · vLLM 0.11.2 (WSL2 Ubuntu 24.04, torch 2.9.0) ·
Windows 10 · commit `1859627`. HF numbers from a 2026-08-12 sweep; vLLM serving numbers
for FP16/AWQ/GPTQ carried over from 2026-07-19 (checkpoints unchanged, so still valid).
**Quality metric:** PPL (wikitext-2) + MMLU-tiny delta vs FP16 — Qwen3-1.7B is text-only,
so the CER column that a VLM version of this table would carry becomes PPL/MMLU here.

## The table

| Config | Decode TPS (HF) | Decode TPS (vLLM) | VRAM GB (HF) | PPL Δ vs FP16 | MMLU Δ | Serving |
|---|---|---|---|---|---|---|
| FP16 baseline | 20.9 | 56.1 | 3.28 | — (22.45) | — (0.548) | ✅ vLLM |
| llmcompressor AWQ-W4A16 | 4.2 | **57.9** | 1.30 | **+15.36** (37.81) | −0.032 | ✅ vLLM (`compressed-tensors`, Marlin) |
| llmcompressor GPTQ-W8A8 | 2.8 | 49.3 | 1.95 | +0.27 (22.72) | +0.008 | ✅ vLLM (`compressed-tensors`) |
| modelopt FP8 | **can't execute**\* | not yet verified† | 1.95 | n/a\* | n/a\* | ⚠️ export fixed, vLLM check blocked (Finding 3) |
| modelopt NVFP4 | n/a | n/a | n/a | n/a | n/a | ❌ needs Blackwell (sm_100+); not attempted |
| modelopt → TRT-LLM | n/a | n/a | n/a | n/a | n/a | ❌ no supported TRT-LLM path on this Windows/WSL setup |
| torchao int4wo | **16.7** | — | **1.47** | +14.02 (36.46) | **−0.132** | HF only (packing-format fix, Finding 5) |
| AWQ-W4A16 + TurboQuant KV (K3V2) | 4.3 | — | 1.30 | +15.40 (37.85)‡ | −0.024‡ | HF only (TurboQuant not wired into vLLM) |

vLLM runs (FP16/AWQ/GPTQ): 0.11.2 in WSL2 Ubuntu 24.04, eager mode (CUDA graph capture
disabled — engine init with graphs OOMed the memory-capped VM; eager slightly
*understates* vLLM TPS, so the serving-vs-eager gap is, if anything, larger than shown).
TTFT: fp16 20.8ms, W4A16 17.7ms, W8A8 35.4ms.

\* **modelopt FP8 now genuinely quantizes — HF eager just can't run it.** After the
export fix (Finding 3), the checkpoint contains real `Float8_e4m3fn` weight tensors +
`hf_quant_config.json` (confirmed: resident VRAM dropped to 1.95 GB, ~41% below fp16).
But plain `AutoModelForCausalLM.from_pretrained` + HF eager has no FP8 compute kernel, so
every forward pass throws `RuntimeError: expected mat1 and mat2 to have the same dtype,
but got: struct c10::BFloat16 != struct c10::Float8_e4m3fn` — PPL/MMLU/TPS are
structurally unmeasurable on this runtime, not degraded. That's expected: FP8 execution
lives in serving engines (vLLM, TensorRT-LLM), not vanilla HF eager. The VRAM number is
real and reported; the compute numbers require the vLLM row.

† **vLLM verification of the FP8 export is blocked, not skipped — the WSL2 disk backing
`vllm-env` has real ext4 corruption** (`dmesg`: "Detected aborted journal", filesystem
remounted read-only), discovered mid-bench. Root cause: unclear, likely an improper
shutdown during this or an earlier session's interim TensorRT-LLM work in the same WSL
distro. Fix is `e2fsck -f` (or a clean venv rebuild) — deliberately **not** run
unattended, since a filesystem repair on a disk holding other in-progress work
(TRT-LLM experiments — see `~/.wslconfig` history) is a call for whoever owns that work,
not something to do silently. FP16/AWQ/GPTQ vLLM numbers above predate the corruption and
are unaffected (confirmed the underlying checkpoints are unchanged since that measurement).

‡ Teacher-forced PPL can't see KV-cache quantization (the cache is never read back during
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
   redundancy to absorb 4-bit weight error — consistent with the general observation that
   aggressive weight quantization punishes small models hardest. A 512-sample calibration
   re-run was attempted to test whether more calibration data closes the gap; killed after
   ~2 hours when per-layer smoothing time grew ~50x with the 4x sample count (a real
   scaling cost of AWQ's per-layer scale search, not a bug) — not worth the GPU time for
   this leaderboard. Open question, not answered here: does 512-sample AWQ recover some
   of this quality? If you need 4-bit at this scale, measure calibration size before
   shipping rather than assuming 128 samples (a common default) is enough.
2. **W8A8 is the quality-safe choice: +0.27 PPL, MMLU within noise.** GPTQ-W8A8 is
   near-lossless on both metrics at 60% of fp16's VRAM (1.95 vs 3.28 GB resident).
3. **This repo's modelopt export was broken at both ends — now fixed and verified on
   the HF side.** `mtq.quantize` + plain `save_pretrained` stored simulated-quant weights
   with no quantizer metadata: HF silently loaded plain bf16 (bit-identical PPL was the
   original tell), and vLLM's loader refused outright (`Cannot find the config file for
   modelopt`). Fixed by routing `ModelOptQuantizer.save` through
   `modelopt.torch.export.export_hf_checkpoint`. Re-quantized and confirmed: the
   checkpoint now ships `hf_quant_config.json` + real `Float8_e4m3fn` tensors, and HF
   correctly identifies it can't execute FP8 (see \* above) rather than silently lying
   about it. vLLM-side confirmation (the actual point of the fix) is blocked by the WSL
   corruption in finding †, above.
4. **The same checkpoint is 14x slower than fp16 in HF eager and *faster* than fp16 in
   vLLM.** AWQ-W4A16: 4.2 TPS on HF eager (per-op dequant, no fused kernels) vs 57.9 TPS
   under vLLM's Marlin kernels — edging out even fp16's 56.1. Never benchmark a
   quantized checkpoint on an eager runtime and conclude the format is slow; runtime
   kernels, not the format, decide the speed story. Memory savings, by contrast, are
   real everywhere (1.30 GB vs 3.28 GB resident on HF).
5. **torchao int4wo was stack-gated, not hardware-gated — now fixed.** torchao 0.17's
   v2-default int4 packing formats (`PLAIN`, `PRESHUFFLED`) both require an `mslk>=1.0.0`
   kernel package with no installable release. Probed every `Int4PackingFormat` on this
   GPU/stack; only `TILE_PACKED_TO_4D` (the classic tinygemm kernel) actually runs. Pinned
   it explicitly (`src/runtimes/hf/hf_runtime.py`, `src/quantization/torch_ao.py`) and
   re-verified: real numbers now (16.7 TPS, 1.47 GB, PPL 36.46). It's also the **worst
   quality result in the table** (MMLU −0.132, steeper than AWQ-W4A16's −0.032 at a
   similar bit-width) — torchao's `int4wo` is calibration-free round-to-nearest, with no
   activation-aware scale search like AWQ's. Cheapest int4 path here, and it shows in the
   quality column, not just the compression ratio.

## Reproduce

```bash
# quantize the local checkpoints
python quantize.py -c config/quantize/qwen3_1_7b/llmc_awq_w4a16.yaml
python quantize.py -c config/quantize/qwen3_1_7b/llmc_gptq_w8a8.yaml
python quantize.py -c config/quantize/qwen3_1_7b/modelopt_ptq_fp8.yaml

# HF-side table numbers (Windows env)
python benchmark.py -c config/benchmark/qwen3_1_7b_leaderboard.yaml

# vLLM-side serving numbers (separate env — WSL here)
python scripts/vllm_bench.py --model Qwen/Qwen3-1.7B
python scripts/vllm_bench.py --model outputs/qwen3-1.7b/awq-w4a16/Qwen3-1.7B-llm_compressor-awq-W4A16 --quantization compressed-tensors
python scripts/vllm_bench.py --model outputs/qwen3-1.7b/modelopt-fp8/Qwen3-1.7B-modelopt-ptq-FP8 --quantization modelopt
# ... one line of JSON per run -> results/qwen3-1.7b-leaderboard/vllm_serving.jsonl

# consolidated W&B comparison view (table + bar charts, one run)
python scripts/wandb_leaderboard.py --dir results/qwen3-1.7b-leaderboard
```

Raw data: `results/qwen3-1.7b-leaderboard/` (per-model JSON + comparison summary with
full environment provenance).

## Open items

- **modelopt FP8 vLLM verification** — blocked on WSL disk repair (`e2fsck -f` on the
  Ubuntu-24.04 distro, or a clean `vllm-env` rebuild in a fresh distro). Once resolved:
  `python scripts/vllm_bench.py --model outputs/qwen3-1.7b/modelopt-fp8/Qwen3-1.7B-modelopt-ptq-FP8 --quantization modelopt`.
- **AWQ-W4A16 at 512-sample calibration** — does more calibration data close the +15.4
  PPL gap? Genuinely unknown; the attempt that would answer this was killed for cost, not
  because it failed.
