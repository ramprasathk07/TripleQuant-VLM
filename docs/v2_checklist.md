# v2 Checklist — what's left to call this project complete

Audited against the code on 2026-08-12, not from memory. Each item says what's missing,
why it matters, and roughly what it costs. Ordered by value-per-effort within each tier.

**Legend:** 🔴 blocks the project being "complete" · 🟡 real value, not blocking ·
🔵 nice-to-have / research · ⛔ needs hardware we don't have

---

## Tier 1 — Correctness & completeness gaps (finish these first)

### 🔴 1. `logit_kl` and `token_agree` are still stubs
`benchmark.py:206` returns `{"skipped": "requires baseline wiring (not yet implemented)"}`.
Both functions **exist and work** in `src/evaluation/eval_llm.py` — what's missing is the
wiring: load the baseline runtime, capture logits/outputs on a fixed prompt set, pass them
in. Config already accepts them as valid metric names, so requesting them today silently
yields nothing.

These are the *right* metrics for weight quantization (paired, per-token, far more
sensitive than MMLU — see `failure_cases.md` #9). Highest correctness-per-hour item on
this list. **~1 day.**

### 🔴 2. Serving metrics: 3 of 4 phases unbuilt
`notes/serving_sla_metrics.md` scopes a full production-SLA harness. Only **Phase 0**
(`runner.py`, OOM-safe orchestration) exists. `src/evaluation/performance/` contains one
file where the plan calls for six.

Missing, in the plan's own order:
- **Phase 0.5** — TTFT-vs-context in `context.py`; use `prompt_lens × output_lens` instead
  of one fixed prompt; compose E2E + request throughput from data already collected.
- **Phase 1** — `records.py` (`RequestRecord`, percentiles) + `streaming.py`
  (`stream_hf`/`stream_vllm`). Gets true **ITL distribution** and kills the current
  two-call TTFT/TPOT subtraction noise.
- **Phase 2** — `load_harness.py` + `sla.py`: async open-loop load with Poisson arrivals,
  **goodput**, QPS sweep, saturation knee. This is the plan's stated deliverable.
- **Phase 3** — schema/reporting wiring for the new metrics.

Why it matters: every latency number in the repo today is **single-request, closed-loop**.
That's the regime that says least about production. Goodput under load is what serving
teams actually gate on, and it's the gap between "I benchmarked a model" and "I benchmarked
a serving system." **~1-2 weeks.**

### 🔴 3. Two model families have never been run — ever
| Family | Configs | Status |
|---|---|---|
| `qwen3_4b_thinking` | 7 quantize + 1 benchmark | schema-valid, **never executed** |
| `hunyuan_ocr` | 5 quantize + 1 benchmark | schema-valid, needs its own pinned-transformers env |
| `qwen2_5_vl_3b` | 5 quantize | only AWQ-W4A16 ever run (CER 0.173) |

A config that has never run is a hypothesis, not a feature. The 4B family also tests
whether the "+15.4 PPL on 4-bit" finding is small-model-specific — it should shrink at 4B,
and that's a real result either way. **~1 day for 4B; hunyuan needs env work first.**

### 🟡 4. Weight-quant quality vs context length — completely unmeasured
PPL uses fixed-size wikitext chunks; MMLU prompts are short. So "does AWQ degrade worse at
16K than at 512?" is unanswered. We measured this axis for the *KV* codec and it produced
the v1's biggest correction (`failure_cases.md` #10) — the same blind spot exists for
weight quantization and nobody has looked. **~1 day** using the existing sweep machinery.

---

## Tier 2 — Making the results stronger

### 🟡 5. Per-channel key quantization (KIVI-style)
The identified root cause of TurboQuant's low-bit quality collapse: keys are quantized
per-vector with one global codebook, but transformer keys have large-magnitude outlier
*channels*. Fix is per-channel keys / per-token values.

This is the single highest-value algorithmic item in the repo. It would move K3V2 from
"unusable at 19% agreement" toward shippable — the difference between 1.73x and 4.8x KV
compression at quality. A quantizer redesign, not a wiring change. **~1-2 weeks.**

### 🟡 6. TurboQuant Triton kernels
Every TQ latency number here (4.3 tok/s vs fp16's 21.5) is the **unfused PyTorch
reference** — 5-10x off what a fused kernel should do. The memory win is banked; the
latency story is unwritten. Designs exist in `notes/turboquant.md` §5; ordered ramp in
[`kernel_learning_path.md`](kernel_learning_path.md) (Level 3 → Level 5). **~3-4 weeks**
for the fused decode kernel, less for the score kernels alone.

### 🟡 7. AWQ calibration-size question
Does 512-sample calibration recover any of the +15.4 PPL that 128 samples cost? The run
that would answer it was killed at ~2h when per-layer smoothing time scaled **~50x** with
4x the samples (a real cost characteristic of AWQ's per-layer scale search, not a bug).
Needs a wider timeout or a smaller model. Genuinely open. **~half a day of GPU time.**

### 🟡 8. TurboQuant × vLLM
Not integrated. A draft adapter existed and was removed in cleanup (imported a module path
that never resolved). Real integration means implementing a vLLM attention backend —
substantial. Without it, TQ can't be measured on the runtime where quantization actually
performs, which blog #1 shows is *the* thing that matters. **~2-3 weeks.**

---

## Tier 3 — Coverage and polish

### 🔵 9. Model zoo breadth
Everything meaningful was measured on Qwen3-1.7B. The findings that likely **don't**
generalize without checking: "+15.4 PPL from 4-bit" (small-model-specific?), the K4V4→K8V8
cliff position (architecture-dependent?). Llama-3-8B, Mistral-7B, Phi, Gemma would test
this. Mostly GPU time, little new code.

### 🔵 10. Downstream evals beyond MMLU
`gsm8k`, `ceval`, `humaneval`, `aime` **are** wired (`eval_tasks.py`) but never included in
a published sweep. Generative evals may show quantization damage that multiple-choice
accuracy structurally cannot — which is exactly the lesson of `failure_cases.md` #9.
Cheap: config change plus GPU time.

### 🔵 11. Auto-generated compatibility matrix
The README's quantizer × runtime table is hand-maintained and will drift. It could be
derived from actual load attempts — the harness already records `error_on_load`.

### 🔵 12. Advisor CLI
`report.py` computes BEST-FOR verdicts; a `triplequant advise --vram 12 --priority latency`
front-end would make the decision matrix executable. Small, high-demo-value.

---

## Tier 4 — Hardware-blocked (not fixable here)

### ⛔ 13. FP8 / NVFP4 serving numbers
Both checkpoints export **correctly** — verified. vLLM rejects FP8 on this GPU with
`Minimum capability: 89. Current capability: 86.` (needs Ada sm_89+; NVFP4 needs
Blackwell). Nothing to fix in the repo; needs different silicon. One cloud GPU-hour would
close three table rows.

### ⛔ 14. TensorRT-LLM engine
No supported path on this Windows/WSL setup. Would also need Ada+ to be worth doing.

---

## Suggested v2 milestone

If the goal is "complete the project" rather than "add features," the minimum defensible
v2 is:

1. **Wire `logit_kl` + `token_agree`** (item 1) — closes a stub that's been advertised in
   the config schema since v0.
2. **Serving Phase 1 + 2** (item 2) — turns single-request latency into real serving
   metrics with goodput. This is the biggest credibility gap.
3. **Run the 4B and VL families** (item 3) — stops shipping untested configs, and tests
   whether the headline 4-bit finding is scale-specific.
4. **Weight-quant vs context sweep** (item 4) — closes the same blind spot that produced
   v1's largest correction.

That's roughly **3-4 weeks** and would leave nothing in the repo that's claimed but
unmeasured. Items 5 and 6 (per-channel keys, fused kernels) are the research/engineering
follow-ons worth doing after, and are individually bigger than everything above combined.

---

*Audited against the tree at commit `d846202`. Sources: `notes/serving_sla_metrics.md`
(phase plan), `benchmark.py:206` (stubs), `config/` (never-run configs),
`docs/failure_cases.md` (open root causes).*
