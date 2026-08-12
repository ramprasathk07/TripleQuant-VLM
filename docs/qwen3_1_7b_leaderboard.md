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

| Config | Decode TPS (HF) | Decode TPS (vLLM) | VRAM GB (HF) | PPL Δ vs FP16 | MMLU Δ (paired p) | Serving |
|---|---|---|---|---|---|---|
| FP16 baseline | 21.5 | 56.1 | 3.28 | — (22.45) | — (0.5337, 427/800) | ✅ vLLM |
| llmcompressor AWQ-W4A16 | 4.3 | **57.9** | 1.30 | **+15.36** (37.81) | −1.1pp (p=0.53, ns) | ✅ vLLM (`compressed-tensors`, Marlin) |
| llmcompressor GPTQ-W8A8 | 2.6 | 49.3 | 1.95 | +0.27 (22.72) | +0.9pp (p=0.19, ns) | ✅ vLLM (`compressed-tensors`) |
| modelopt FP8 | **can't execute**\* | **can't execute**† | 1.95 | n/a\* | n/a\* | ❌ needs Ada+ (sm_89); export verified correct |
| modelopt NVFP4 | n/a | n/a | n/a | n/a | n/a | ❌ needs Blackwell (sm_100+); not attempted |
| modelopt → TRT-LLM | n/a | n/a | n/a | n/a | n/a | ❌ no supported TRT-LLM path on this Windows/WSL setup |
| torchao int4wo | **17.3** | — | **1.47** | +14.02 (36.46) | **−9.9pp (p<0.0001)** | HF only (packing-format fix, Finding 5) |
| AWQ-W4A16 + TurboQuant KV (K3V2) | 4.3 | — | 1.30 | +15.40 (37.85)‡ | −0.1pp (p=1.00, ns) | HF only (TurboQuant not wired into vLLM) |

**"ns" = not significant.** MMLU here is n=800 questions (5 subjects × 200, two subjects
capped at their 100-row test splits) with a paired McNemar exact test against the FP16
baseline — every model answers identical questions, so the right comparison is over the
*discordant* questions, not two independent accuracy figures. Reproduce with
`python scripts/mmlu_significance.py --dir results/qwen3-1.7b-leaderboard`. **Only
torchao int4wo's regression is statistically real**; every other MMLU delta in this
table is noise and must not be reported as a quality finding (see Finding 6).

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

† **The FP8 export fix is verified — by the error message changing.** vLLM was re-run
against the fixed checkpoint in a clean `vllm-env2` (vLLM 0.11.2, WSL2). Before the fix
it failed with `Cannot find the config file for modelopt` — a *repo* bug, the export was
missing its quant metadata entirely. After the fix it fails with:

```
The quantization method modelopt is not supported for the current GPU.
Minimum capability: 89. Current capability: 86.
```

vLLM now parses the checkpoint, reads its `hf_quant_config.json`, and gets all the way to
the hardware-capability gate before stopping. That is the export working correctly: FP8
serving needs compute capability **sm_89 (Ada Lovelace)** and this RTX 3060 is **sm_86
(Ampere)**. The row is unservable *here* for the same reason as NVFP4 — a hardware floor,
not a defect. On an Ada/Hopper card this checkpoint should serve as-is; that's untested by
us and stated as an expectation, not a result. (An earlier attempt hit ext4 journal
corruption on the original `vllm-env` disk; it cleared on remount, and the rebuild in a
fresh venv sidestepped it entirely.)

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

1. **W4A16 costs +15.4 PPL on a 1.7B model — but its MMLU drop is *not* significant.**
   AWQ-W4A16 at 128 calibration samples took Qwen3-1.7B from 22.45 to 37.81 PPL (+68%),
   a large, unambiguous quality cost on the metric that can resolve it. MMLU tells a
   different story: −1.1pp, p=0.53 — indistinguishable from noise even at n=800. The
   discordant breakdown explains why: AWQ changes the answer on **163 of 800 questions**
   (86 baseline-right→model-wrong, 77 the other way) — it perturbs the model heavily but
   *symmetrically*, so net accuracy barely moves. Multiple-choice accuracy is simply a
   blunter instrument than perplexity for detecting this kind of damage. An earlier
   version of this table reported "−3.2pp MMLU" as a finding at n=250; that was noise
   (see Finding 6). A 512-sample calibration re-run was attempted to test whether more
   calibration data closes the PPL gap; killed after ~2h when per-layer smoothing time
   grew ~50x with the 4x sample count (a real scaling cost of AWQ's per-layer scale
   search, not a bug). Still open.
2. **W8A8 is the quality-safe choice: +0.27 PPL, MMLU flat.** GPTQ-W8A8 is near-lossless
   at 60% of fp16's VRAM (1.95 vs 3.28 GB resident). Its +0.9pp MMLU is not significant
   (p=0.19) and, notably, it only disturbs **21 of 800 questions** — an order of
   magnitude fewer than AWQ's 163, which is what "near-lossless" looks like mechanically.
3. **This repo's modelopt export was broken at both ends — now fixed and verified on
   the HF side.** `mtq.quantize` + plain `save_pretrained` stored simulated-quant weights
   with no quantizer metadata: HF silently loaded plain bf16 (bit-identical PPL was the
   original tell), and vLLM's loader refused outright (`Cannot find the config file for
   modelopt`). Fixed by routing `ModelOptQuantizer.save` through
   `modelopt.torch.export.export_hf_checkpoint`. Re-quantized and confirmed: the
   checkpoint now ships `hf_quant_config.json` + real `Float8_e4m3fn` tensors, and HF
   correctly identifies it can't execute FP8 (see \* above) rather than silently lying
   about it. **vLLM-side confirmation now done too, and the error message is the
   receipt**: the loader went from "can't find the config" (repo's fault) to "needs
   sm_89, you have sm_86" (hardware floor) — see † above. The fix is verified end to end;
   the row stays unservable on this GPU for a reason that has nothing to do with the
   repo.
4. **The same checkpoint is 13x slower than fp16 in HF eager and *faster* than fp16 in
   vLLM.** AWQ-W4A16: 4.3 TPS on HF eager (per-op dequant, no fused kernels) vs 57.9 TPS
   under vLLM's Marlin kernels — edging out even fp16's 56.1. Never benchmark a
   quantized checkpoint on an eager runtime and conclude the format is slow; runtime
   kernels, not the format, decide the speed story. Memory savings, by contrast, are
   real everywhere (1.30 GB vs 3.28 GB resident on HF). This gap is the concrete
   motivation for the kernel work laid out in
   [`docs/kernel_learning_path.md`](kernel_learning_path.md).
5. **torchao int4wo was stack-gated, not hardware-gated — now fixed.** torchao 0.17's
   v2-default int4 packing formats (`PLAIN`, `PRESHUFFLED`) both require an `mslk>=1.0.0`
   kernel package with no installable release. Probed every `Int4PackingFormat` on this
   GPU/stack; only `TILE_PACKED_TO_4D` (the classic tinygemm kernel) actually runs. Pinned
   it explicitly (`src/runtimes/hf/hf_runtime.py`, `src/quantization/torch_ao.py`) and
   re-verified: real numbers now (17.3 TPS, 1.47 GB, PPL 36.46). It's also the **only row
   with a statistically real MMLU regression** (−9.9pp, p<0.0001) — torchao's `int4wo` is
   calibration-free round-to-nearest, with no activation-aware scale search like AWQ's.
   The discordance is asymmetric (172 questions broken vs 93 fixed), which is what real
   degradation looks like next to AWQ's symmetric 86/77 churn. Interesting tension: it's
   the *fastest* quantized row in HF eager (17.3 TPS vs AWQ's 4.3 — no per-group dequant
   bookkeeping) and the worst quality. Cheap in every sense.
6. **Most of the quality deltas in the first version of this table were noise, including
   one I reported as a finding.** The original MMLU used n=250 (5 subjects × 50), where
   the binomial SE is ±3.1pp — so a "+0.8pp improvement" was literally 2 questions out of
   250 flipping, and even the "−3.2pp AWQ regression" I wrote up was ~1 SE. Two fixes
   landed: `eval_mmlu_tiny` now returns counts, stderr, and per-question outcomes instead
   of a bare float (a bare accuracy hides its own sample size, which is what made this
   invisible), and `scripts/mmlu_significance.py` runs the paired McNemar exact test that
   this comparison always needed. Re-run at n=800: only int4wo survives. **If a quantized
   model appears to beat its own baseline on a benchmark, that is the null hypothesis
   presenting itself, not a discovery** — quantization removes information; it does not
   add capability.

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

- ~~modelopt FP8 vLLM verification~~ — **done.** Export confirmed correct; the row is
  blocked by a hardware floor (sm_89 required, sm_86 available), not by the repo. Would
  need an Ada/Hopper/Blackwell GPU to produce a real serving number.
- **AWQ-W4A16 at 512-sample calibration** — does more calibration data close the +15.4
  PPL gap? Genuinely unknown; the attempt that would answer this was killed for cost, not
  because it failed.
- **Kernel work** — the 13x HF-eager-vs-vLLM gap on the *same* AWQ checkpoint is the
  clearest optimization target this table exposes. Ordered ramp:
  [`docs/kernel_learning_path.md`](kernel_learning_path.md).
