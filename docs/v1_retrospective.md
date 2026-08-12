# v1.0 Close — what this was for, and what it actually found

Written at the v1 freeze. The other docs report *results*; this one reports the
*project*: why it started, what the original goal turned out to be worth, and which of
the findings would survive someone senior poking at them.

---

## Why this started

First commit, 2026-05-20, described it as:

> *"Production-grade AWQ/GPTQ weight quantization + benchmarking for Vision-Language
> Models (e.g. Qwen2.5-VL), featuring vLLM serving, CER/WER OCR evaluation, and
> structured experiment tracking."*

The motivating question was practical, not academic: **given one consumer GPU and a model
you want to run on it, which quantization method should you actually pick?** Every
individual method has a paper and a README claiming it works. Almost nothing compares
them under one harness, on one machine, with the failures included.

Two things shifted during the build:
- **VLM → LLM as the primary test bed.** Qwen2.5-VL work happened (OCR CER 0.173 on
  AWQ-W4A16) but Qwen3-1.7B became the workhorse: faster iteration, and text metrics
  (PPL/MMLU) that are far better understood than OCR CER when the goal is *comparing
  methods* rather than shipping an OCR product.
- **Weight quantization → weight *and* KV-cache quantization.** Adding TurboQuant was the
  single biggest scope change, and it's what turned the project from "wrap three
  libraries" into something with an original result.

---

## What the project is actually worth, honestly

**The framework itself is table stakes.** A registry, a config schema, and three backend
adapters is a few weeks of unglamorous work. It is not the interesting part, and nobody
should be impressed by it in isolation.

**The interesting part is the measurements that contradict the intuitive answer.** Four
of those, in rough order of how much they'd survive scrutiny:

### 1. The runtime decides quantization's speed, not the format
The *same* AWQ-W4A16 checkpoint: **4.3 tok/s under HF eager, 57.9 tok/s under vLLM** —
faster than fp16's 56.1. A 13x swing with the weights held constant. Anyone
benchmarking a quantized model on an eager runtime and concluding "int4 is slow" has
measured their runtime, not their format. This is the single most portable lesson here
and it cost nothing to find beyond running both.

### 2. 4-bit weights on a small model are expensive, and MMLU can't see it
AWQ-W4A16 on Qwen3-1.7B: **+15.4 PPL** (22.45 → 37.81, +68%). But MMLU: −1.1pp,
p=0.53 — statistically indistinguishable from noise at n=800. The reason is visible in
the paired data: AWQ changes **163 of 800** MMLU answers, but *symmetrically* (86 broken,
77 fixed), so accuracy barely moves while the model's actual behaviour has shifted
substantially. Perplexity sees it; multiple-choice accuracy doesn't. Choose the metric
that can resolve the thing you're claiming.

### 3. TurboQuant's usable win is at K8V8, not the aggressive setting
4x context on the same GPU (16,384 vs fp16's 4,096) **with 0.918 top-1 agreement at 16K**
at K8V8. The aggressive K3V2 setting reaches identical capacity at 0.193 agreement — its
extra compression buys nothing and costs nearly all fidelity. The quality cliff sits
between K4V4 and K8V8, and agreement does *not* decay with context at any bit-width
(K3V2 is equally poor at 512), so the constraint is the bit budget, not long context.
Root cause: per-vector key quantization with no outlier-channel handling; the known fix
is per-channel keys (KIVI-style), which is a quantizer redesign and not done here.

### 4. Cheap int4 is cheap in every sense
torchao `int4wo` is calibration-free round-to-nearest — the *fastest* quantized row in HF
eager (17.3 tok/s vs AWQ's 4.3) and the **only** row with a statistically real MMLU
regression (−9.9pp, p<0.0001). Its 265 changed answers are *asymmetric* (172 broken / 93
fixed), which is what genuine degradation looks like beside AWQ's symmetric churn. Skip
the calibrated scale search, pay for it in quality.

---

## What I got wrong, and how it was caught

Four claims in earlier versions of these docs were wrong. All four were caught by
re-deriving numbers from raw JSON rather than trusting the prose — which is the actual
methodological finding of this project.

| Wrong claim | Reality | Caught by |
|---|---|---|
| "torchao int8wo reaches 8,192 context" | 4,096 — same as fp16 | Reading `max_fit_tokens` instead of eyeballing a chart |
| "AWQ costs −3.2pp MMLU" | −1.1pp, p=0.53 — noise | Reporting n and running a paired test |
| "quantization improved MMLU/PPL" (several rows) | 2-3 questions of 250 flipping | Same |
| "TurboQuant K3V2 gives 4x usable context" | Capacity yes, usability no — K8V8 is the shippable setting | Sweeping quality across the *same* context range as the capacity claim |

The common structure: **a metric that couldn't see what it was being used to claim.**
Teacher-forced PPL can't observe KV-cache error ([#4](failure_cases.md)). Bare accuracy
hides its own sample size ([#9](failure_cases.md)). A memory probe says nothing about
output quality ([#10](failure_cases.md)). Three instances, three different layers, one
mistake.

If there's one habit worth carrying out of this project: **before reporting a number, ask
what would have to be true for it to be wrong, then check that specific thing.**

---

## Scope: what's real, what's hardware-blocked, what's untouched

**Verified working:** llm_compressor (AWQ/GPTQ/PTQ/SmoothQuant), ModelOpt (export fixed
this cycle — see below), torchao (int8wo, int4wo after a packing-format fix), TurboQuant
KV compression, HF + vLLM runtimes, crash-safe benchmark harness, report generator, W&B.

**Hardware-blocked, checkpoints correct:** ModelOpt FP8 needs sm_89 (Ada); NVFP4 needs
Blackwell. The FP8 export bug was real and is fixed — `save_pretrained` was storing
simulated-quant weights with no quantizer metadata, so HF silently loaded bf16 and vLLM
refused the checkpoint outright. Now routed through `export_hf_checkpoint`. The fix is
verified *by the error changing*: vLLM went from "cannot find the config file" (our bug)
to "minimum capability 89, current 86" (physics).

**Not done, stated plainly:** TurboQuant's fused Triton kernels (designed in
`notes/turboquant.md` §5, ramp in [`kernel_learning_path.md`](kernel_learning_path.md) —
every TQ latency number here is the unfused reference); TurboQuant × vLLM; per-channel
key quantization; TensorRT-LLM; whether 512-sample AWQ calibration recovers the PPL gap
(the run was killed at ~2h when per-layer cost scaled ~50x with sample count); any
context-length quality data for *weight* quantization.

---

## If someone picks this up next

In value order:

1. **Per-channel key quantization for TurboQuant.** The identified fix for the one
   quality problem blocking aggressive bit-widths. Would move K3V2 from unusable toward
   usable, which is the difference between 1.73x and 4.8x KV compression at ship quality.
2. **The fused decode kernel.** TurboQuant's memory win is banked; the latency cost (4.3
   vs fp16's 21.5 tok/s) is entirely unfused PyTorch. `kernel_learning_path.md` Level 5.
3. **Weight-quant quality vs context length.** Genuinely unmeasured here — PPL uses
   fixed-size chunks and MMLU prompts are short. Does AWQ degrade worse at 16K than at
   512? Nobody in this repo knows.
4. **Re-run the whole thing on an Ada+ GPU.** Three table rows are hardware-blocked
   rather than unknown, and FP8 is the most deployable format of the lot.

---

*Numbers: Qwen3-1.7B, RTX 3060 12GB, driver 610.62 / CUDA 12.8 / torch 2.8.0 /
transformers 4.55.4; vLLM 0.11.2 in WSL2. Full tables in
[`qwen3_1_7b_leaderboard.md`](qwen3_1_7b_leaderboard.md) and
[`benchmark_report.md`](benchmark_report.md); every failure in
[`failure_cases.md`](failure_cases.md); raw JSON under `results/`.*
