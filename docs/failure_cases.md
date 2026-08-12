# Failure Cases & Known Limitations

Honest documentation of what doesn't work, what's slower than it should be, and what
looked broken but wasn't. Each entry: symptom, root cause, fix or current status,
evidence. Kept up to date as the project moves — an entry closed by a fix stays here
with its resolution, not deleted, so the debugging trail survives.

---

## 1. Windows WDDM silently oversubscribes VRAM (fixed 2026-07-17)

**Symptom:** `measure_ctx_sweep`/`measure_max_context` (`src/runtimes/hf/hf_runtime.py`)
reported `peak_vram_mb: 14174` with `fits: true` at 8192-token context, on a 12288 MB
RTX 3060. Physically impossible for a real CUDA allocation.

**Root cause:** on Windows with the WDDM driver model, a CUDA allocation that exceeds
dedicated VRAM can silently spill into shared system RAM instead of raising
`OutOfMemoryError`. `torch.cuda.max_memory_allocated()` faithfully reports what the
allocator believes it allocated — which can exceed the card's physical memory — while
the forward pass "succeeds," just at PCIe-transfer speed instead of VRAM bandwidth. Any
benchmark that treats "no exception" as "it fits" inherits this lie.

**Fix:** compare peak allocated against `torch.cuda.get_device_properties().total_memory`
(a static hardware fact WDDM's paging can't inflate). Anything past 95% of it is flagged
`oversubscribed: true`, excluded from `max_fit_tokens`, and the sweep stops there rather
than continuing into increasingly-fake territory. Commit `2289023`.

**Evidence:** re-running the same probe after the fix converges `measure_max_context` on
Qwen3-1.7B/RTX3060 to 6912 tokens instead of continuing past the physical boundary.
`docs/benchmark_report.md`'s context-length chart is built entirely on the corrected
measurement.

**Takeaway for anyone benchmarking on Windows:** don't trust a "fits" signal from exception
absence alone. Cross-check against `torch.cuda.get_device_properties().total_memory`.

---

## 2. TurboQuant's default K3V2 has weak next-token agreement at low bit-widths

**Symptom:** measured next-token agreement with the FP16 baseline (at the compressed-tail
KV positions, i.e. tokens old enough to have left the exact ring buffer): K2V2 and K3V2
both score 9-19% across context lengths 512-4096. That's barely above what a large-vocab
model would score by chance on a hard prompt.

**Root cause:** the key quantizer treats each key vector independently with a single
global codebook and no per-channel outlier handling. Transformer key vectors have a small
number of large-magnitude *channels* (this is the same outlier-channel phenomenon
SmoothQuant/AWQ correct for in weights) — a uniform per-vector codebook wastes bits on
channels that don't need precision and starves the channels that do. Value vectors don't
have this pathology as severely, which is why V4 recovers much more agreement than K3->K4
does at matched total bits (see the table in `docs/benchmark_report.md`: K2V2 vs K3V2
score within noise of each other — both are V2 — while K4V4 is a clear step up).

**Status: not fixed, understood.** The real fix is per-channel key quantization (KIVI's
approach: quantize keys per-channel, values per-token) — a quantizer redesign, not a
wiring bug. Verified via the identity test in
[`debugging_turboquant_kv.md`](engineering_notes.md): round-trip
`dequantize(quantize(x))` error falls monotonically with bit-width (K3 0.43 -> K4 0.23 ->
K8 0.02 relative error) — a *correct* codec operating at a tight, uneven budget, not a
bug. A clean K8/V8 generation run (fully coherent output) rules out an integration bug by
construction: if the wiring were broken, high bits wouldn't fix it either.

**Mitigation today:** the pipeline ships the full bit-width sweep rather than hiding it —
`docs/benchmark_report.md`'s tradeoff table lets you pick the point on the curve that
matches your quality bar. K8V8 is near-lossless (~97-98% agreement) at a modest 1.3-1.7x
compression; K3V2 trades quality for a much larger compression ratio (up to 4.07x at
context 4096) and a 4x context-length win (see `docs/benchmark_report.md`'s headline
result) — whether that trade is worth it is workload-dependent, which is the point of
measuring and reporting it rather than picking a "best" default and hiding the rest.

---

## 3. QJL 1-bit residual correction doesn't help in practice

**Symptom:** the QJL correction (`src/turboquant_v1/quantize.py`) is designed to make the
inner-product estimate unbiased, but empirically degrades generation quality compared to
MSE-only quantization at the same bit budget.

**Root cause, precisely:** the QJL-corrected estimator and dequantize-then-matmul are
**algebraically identical** for a fixed random projection `S` —
`q @ (k_mse + x_qjl).T == (q @ k_mse.T) + (q @ S.T) @ signs.T * scale`. QJL's theoretical
benefit is variance reduction *in expectation over random `S`*, not a per-call accuracy
gain for one fixed draw of `S`. In this implementation `S` is fixed once at init (for
speed — resampling per call would cost more than it's worth), so the theoretical benefit
never materializes, and the extra 1-bit-per-coordinate storage buys nothing.

**Status:** MSE-only ships as the default (`use_qjl: false`); QJL is exposed via
`--use-qjl` for experimentation, not recommended. Documented in the README's TurboQuant
section and re-derived independently in
[`debugging_turboquant_kv.md`](engineering_notes.md) Case in Phase 3.2 — a fix hypothesis
that looked plausible, was measured numerically before being built, and turned out to be
a dead end. Kept as a worked example of testing a fix's premise before implementing it.

---

## 4. Teacher-forced perplexity cannot see KV-cache quantization error

**Symptom:** early benchmark sweeps (through 2026-06-08) showed TurboQuant's PPL and
MMLU-tiny scores as *bit-identical* to the FP16 baseline — `22.449370186003648` to 16
decimal places. That's not "no quality loss," it's the metric being blind.

**Root cause:** perplexity evaluation is teacher-forced — every ground-truth token is fed
back into the model as input for the next position, computed in a single forward pass over
the whole sequence. The KV cache is populated and never *read back* for anything the loss
depends on (each position's logits only need the causal-masked prefix, materialized fresh
each time in one pass). TurboQuant compresses the *cache*; a metric that never exercises
cache reads structurally cannot detect cache compression error, regardless of how lossy
the codec is.

**Fix:** measure quality via actual **generation** — multi-step decode, where each new
token's attention genuinely reads back the (possibly compressed) cache from previous
steps. `measure_bits_accuracy` / the `tq_bits_sweep` metric
(`src/runtimes/hf/hf_runtime.py`) does exactly this: forward passes with
`past_key_values` populated, comparing next-token logits against a cache-free FP16
reference on the same input.

**Takeaway for anyone evaluating KV-cache compression:** if your quality metric doesn't
change when you swap the KV-cache codec, check whether the metric ever reads the cache
before concluding the codec is lossless. This is general to any KV-cache technique (FP8
KV, KIVI, TurboQuant, ...), not specific to this implementation.

---

## 5. FP8 is emulated (slow) on Ampere

**Symptom:** FP8 quantization schemes (`FP8`, `FP8_DYNAMIC`) load and run on the RTX 3060,
but with a real performance penalty compared to native FP8 hardware.

**Root cause:** FP8 E4M3/E5M2 Tensor Core acceleration is Ada Lovelace/Hopper+ only.
Ampere (RTX 30-series, A100) has no native FP8 matrix path — vLLM and torch fall back to
emulated dequant-then-compute, which works correctly but forfeits the format's speed
advantage entirely; the only benefit retained on Ampere is the smaller checkpoint/VRAM
footprint, not throughput.

**Status:** documented hardware floor, not a bug. See the README's Hardware Compatibility
Floors table. Confirmed correct but slow in practice on this project's dev hardware.

**Update (2026-08-12) — vLLM refuses FP8 outright on Ampere, it doesn't even emulate.**
Serving the ModelOpt FP8 checkpoint through vLLM 0.11.2 fails at config validation:

```
The quantization method modelopt is not supported for the current GPU.
Minimum capability: 89. Current capability: 86.
```

So there are two distinct behaviours worth keeping straight: the *HF/torch* path will
happily run FP8 numerics in emulation (slow, works), while *vLLM* gates on compute
capability and declines to load at all. "FP8 is emulated on Ampere" is true of the
framework, not of every runtime. Useful corollary: this error is also how we verified the
ModelOpt export fix (case #3 in `docs/qwen3_1_7b_leaderboard.md`) — reaching a *hardware*
rejection means the checkpoint's metadata parsed correctly, which it previously did not.

---

## 6. NVIDIA ModelOpt's CUDA extension won't JIT on Windows (no MSVC `cl.exe`)

**Symptom:** ModelOpt-backed AWQ quantization runs correctly but falls back to a slow CPU
extension path on Windows dev boxes.

**Root cause:** `modelopt_cuda_ext` JIT-compiles a CUDA extension at import time via
`torch.utils.cpp_extension`, which shells out to an MSVC `cl.exe` host compiler. Without
Visual Studio C++ Build Tools installed, the JIT silently falls back to a pure-Python/CPU
implementation of the same op — correct output, much slower calibration.

**Fix:** install VS C++ Build Tools to enable the fast path. Left undone in this dev
environment (calibration still completes, just slower) — flagged here so it isn't
mistaken for a correctness problem if someone sees a slow AWQ run on Windows.

---

## 7. vLLM needs a separate environment from the quantization stack

**Symptom:** installing `vllm` alongside `llmcompressor`/`modelopt`/`torch` in one
environment routinely produces dependency conflicts (see the README's "Environment
Separation Note").

**Root cause:** vLLM pins its own compatible torch/transformers/CUDA range, which drifts
out of sync with the quantization backends' pins on a different release cadence. Both
ecosystems move fast enough that a single shared environment breaks intermittently.

**Fix:** documented split — quantize and run HF-runtime evaluation in the primary
environment; run the vLLM runtime in a separate, clean environment
(`pip install -e ".[vllm]"` in its own venv, now that this is a proper optional extra
rather than a hard dependency — see the pyproject.toml hygiene pass). Not a bug, a
packaging-ecosystem reality; documented rather than fought.

---

## 8. TurboQuant's fused Triton kernels aren't built yet

**Status:** the PyTorch reference path (`src/turboquant_v1/`) is what every result in this
repo runs on — correct, but 5-10x slower than a fused decode kernel would be, per the
kernel-scope estimate in `notes/kernel_scope.md`. The Triton kernels (MSE score, QJL
score, fused decode) are scoped and designed (math + pseudocode in `notes/turboquant.md`
§5) but not implemented. This means every TPOT/TPS number in `docs/benchmark_report.md`
for the TurboQuant entries reflects the *unfused* reference implementation — the memory
story (context-length capacity) is real and already measured; the latency story has more
headroom than what's shown once/if the kernels land. Ordered learning path for building
them: [`docs/kernel_learning_path.md`](kernel_learning_path.md).

---

## 9. Reporting an accuracy metric without its sample size (fixed 2026-08-12)

**Symptom:** the leaderboard showed several quantized models *beating* the FP16 baseline
on MMLU (+0.8pp, +1.6pp), and one finding was written up asserting AWQ-W4A16 cost −3.2pp
MMLU. Both directions were wrong.

**Root cause:** `eval_mmlu_tiny` returned a bare `correct / total` float. With the
denominator discarded, nothing downstream — not the results JSON, not the comparison
table, not the docs — could tell that n was only 250 (5 subjects × 50, a hardcoded
constant). At n=250 the binomial standard error is ±3.1pp and the 95% CI is ±6.2pp, so:

| Reported delta | What it actually was |
|---|---|
| +0.8pp "improvement" | 2 questions out of 250 |
| +1.6pp "improvement" | 3 questions |
| −3.2pp "regression" (written up as a finding) | 8 questions, ~1 SE |

A quantized model cannot gain capability its base model lacked — an apparent improvement
is the null hypothesis announcing itself. That should have been the tell.

**Fix, three parts:**
1. `eval_mmlu_tiny` now returns `{acc, correct, total, stderr, per_subject,
   per_question}`. A metric that carries its own sample size can't hide this.
2. The question count moved from a hardcoded `_MMLU_Q_PER_SUBJECT = 50` in `benchmark.py`
   to `datasets.mmlu_num_q_per_subject` in the config, and was raised to 200 (n≈800).
3. `scripts/mmlu_significance.py` runs the **paired McNemar exact test** — the correct
   tool, since every model answers identical questions, making the discordant pairs far
   more informative than two independent accuracy figures.

**Result after re-running at n=800:** every MMLU delta except torchao int4wo's (−9.9pp,
p<0.0001) is not significant. AWQ-W4A16 came in at −1.1pp, p=0.53 — the finding it had
generated was retracted. The discordant counts turned out to be the more interesting
signal anyway: AWQ changes 163 of 800 answers but symmetrically (86 broken / 77 fixed),
while int4wo changes 265 asymmetrically (172 / 93). Same headline "4-bit weights", very
different mechanisms.

**Generalizes to:** any accuracy-style eval (MMLU, GSM8K, HumanEval, C-Eval) in this
repo or elsewhere. Report `correct/total` and a CI, not a lone ratio; use a paired test
when the same items are scored by both systems. The same "metric can't see what you
think it sees" failure mode also produced case #4 above (teacher-forced PPL cannot
observe KV-cache quantization) — worth reading the two together.

---

## 10. The headline "4x context" claim had no quality measurement behind it (fixed 2026-08-12)

**Symptom:** the README led with "TurboQuant K3V2 reaches 16,384 tokens — 4x fp16." True
as stated, and materially misleading: it was a pure *memory-capacity* claim presented
where a reader would assume usability.

**Root cause — two separate blind spots:**
1. `measure_ctx_sweep`, which produces the 4x number, records
   `kv_cache_mb / peak_vram_mb / fits`. **There is no accuracy field in it at all.** It
   answers "does the cache fit", never "is the output any good".
2. `measure_bits_accuracy`, the one metric that *does* pair quality with context length,
   stopped at 4,096 — below the 16,384 the claim rested on — and compared only
   `min(64, ...)` positions per point, a ~5pp binomial SE. The apparent context trends
   were smaller than their own error bars: K2V2 *rose* mid-sweep, K4V4 dipped at 2048 and
   recovered. Unreadable.

So the claim's context range and its quality range never overlapped.

**Fix:** `compare_positions` is now configurable (`latency.tq_compare_positions`, default
512 → SE ~1-3pp), `tq_sweep_lengths` extends to 16,384 to cover the claim, and each grid
point carries `top1_stderr`. Then re-run, plus a context sweep at K8V8 to find the
capacity of a *quality-preserving* setting.

**What the measurement actually showed — the claim survived, but pointed at the wrong
configuration:**

| Setting | Max context | Top-1 agreement @16K | KV compression @16K |
|---|---|---|---|
| fp16 | 4,096 | — | 1.0x |
| TurboQuant K3V2 (was showcased) | 16,384 | **0.193** | 4.81x |
| TurboQuant K8V8 | 16,384 | **0.918** | 1.73x |

K8V8 delivers the same 4x capacity *with* quality intact. K3V2's extra compression buys
no additional context here and costs almost all output fidelity. **We had been
showcasing the strictly worse of the two.** Also settled: agreement does not decay with
context at any bit-width (K3V2 is as poor at 512 as at 16K), so the low-bit problem is
the bit budget, not long context.

**Two caveats kept in the docs rather than smoothed over:**
- The sweep varies context length *and* the compared text together (each length compares
  the last N positions of a different-length slice), so per-length wobble is confounded
  and shouldn't be read as a context effect.
- "Max context" is what a single full-prefill forward pass fits in 12 GB, and that peak
  is dominated by the all-position logits tensor rather than the KV cache. Valid as a
  *relative* comparison under an identical probe; not a serving capacity number.

**Lesson:** when a headline pairs two axes ("this much context"), check that the
supporting measurement actually spans both. Here the capacity probe and the quality probe
covered disjoint ranges, and nothing in the pipeline flagged it — the same structural
blind spot as #4 and #9, in a third place.
