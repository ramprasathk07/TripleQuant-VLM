# GPU Kernel Learning Path — where this repo needs Triton/CUDA, easiest first

A ramp, not a spec. For each rung: *what the kernel actually does*, *why this repo
needs it* (grounded in measured numbers from our own benchmarks, not general claims),
*where it plugs in*, and *how you'd know it works*.

**Companion docs — read them for the deep detail, this page for the ordering:**
- [`notes/turboquant.md`](../notes/turboquant.md) §5 — Kernels A/B/C/D with real Triton
  pseudocode and optimization knobs. The *what to write*.
- [`notes/kernel_scope.md`](../notes/kernel_scope.md) — P0–P5 priority table, Triton vs
  TileLang decision tree, TileLang learning path, profiling methodology. The *what to
  pick and why*.
- This page — the *what order to learn it in, and what each thing is for*.

---

## The one number that motivates all of it

From our own leaderboard (`docs/qwen3_1_7b_leaderboard.md`), the **same AWQ-W4A16
checkpoint**:

| Runtime | Decode TPS |
|---|---|
| HF eager (no fused kernels) | 4.3 |
| vLLM (Marlin int4 kernels) | **57.9** |

**13.6x, same weights, same GPU.** The only difference is that vLLM has a kernel that
keeps the weights packed in registers and dequantizes inline, while HF eager
materializes a dequantized bf16 tensor for every matmul and pays a full HBM round-trip
for it. That gap *is* the value of kernel work, measured on this box.

The second motivating number: TurboQuant gets **4x context length** (16,384 vs 4,096
tokens) but runs at 4.3 TPS vs fp16's 21.5. The memory win is real and already banked;
the speed cost is entirely because the KV codec runs as unfused PyTorch ops.

---

## Level 0 — Concepts to hold before writing anything (~2-3 days)

Not code. If these four ideas aren't solid, kernel code is cargo-culting.

**1. The memory hierarchy, and that it is the whole game.**

```
Registers   ~20 TB/s    per-thread, tiny
Shared/SRAM ~10 TB/s    per-block, ~100 KB
L2           ~5 TB/s
HBM (VRAM)  ~360 GB/s   <- RTX 3060. This is the bottleneck.
```
A fused kernel is fast because it reads HBM *once* and does all its work in the top
three tiers. An unfused sequence of PyTorch ops reads and writes HBM between every
step. That's the entire mechanism behind the 13.6x above.

**2. Arithmetic intensity and the roofline.** FLOPs ÷ bytes moved. Decode at batch=1 is
extremely low intensity — one token's worth of math against the *whole* weight matrix
and KV cache. So decode is **memory-bound**: performance is set by bytes moved, not
math done. Corollary that trips people up: at batch=1, int4 weights are faster than
fp16 *not because int4 math is faster* but because there are 4x fewer bytes to move.

**3. GPU execution model.** grid → blocks → warps (32 threads, lockstep) → threads.
Triton works at *block* level: you write code for one block, Triton handles threads
within it. This is why Triton is dramatically easier than raw CUDA for this work.

**4. Why quantized models are often *slower* in eager mode.** Store int4, dequantize to
bf16, matmul in bf16. The dequant is an extra HBM round-trip that fp16 never pays. You
get the memory saving and a speed *penalty* — exactly what our leaderboard shows. The
fix is fusing dequant into the matmul, which is what Marlin does.

*Checkpoint:* explain why our AWQ checkpoint uses 1.30 GB but runs 5x slower than the
3.28 GB fp16 model in HF eager. If that's clear, continue.

**Resources:** Triton tutorials 01–02; the FlashAttention-2 paper's §2 (background) for
the memory-hierarchy framing.

---

## Level 1 — First Triton kernels (~3-4 days)

**What you're doing:** learning the API on problems whose answers you can verify
trivially. No repo integration; pure practice.

| Kernel | New skill |
|---|---|
| Vector add | `program_id`, `tl.arange`, `tl.load/store`, masking for ragged tails |
| Fused elementwise (`x * sigmoid(x)`) | why fusion wins — one HBM read instead of three |
| Row-wise softmax | reductions (`tl.max`, `tl.sum`), numerical stability (subtract max) |
| Matrix transpose | 2D indexing, memory coalescing, bank conflicts |

**Validation habit to build now:** every kernel gets `torch.allclose` against a PyTorch
reference on random inputs, plus `triton.testing.do_bench` against the same. Both, every
time. `notes/kernel_scope.md` §5 has the full gate list this repo expects.

*Why softmax matters later:* the online-softmax trick in Level 5 is a streaming variant
of exactly this. Write the simple one first.

---

## Level 2 — Dequantization kernel ⭐ first genuinely useful one (~1 week)

**What we're doing:** taking packed int4/int8 weights + per-group scales and producing
bf16 — the operation that makes our AWQ checkpoint 13.6x slower in eager mode than under
vLLM.

Concretely, a W4A16 weight is stored as:
- `qweight`: uint8, two 4-bit values per byte
- `scales`: one fp16 per group of 128 weights
- (asymmetric adds `zeros`)

Naive: `unpack → cast → multiply by scale → write bf16 tensor to HBM → matmul reads it
back`. Fused: unpack and scale *inside* the matmul's inner loop, so the bf16 tensor is
never written at all.

**Build order:**
1. Standalone dequant kernel: packed int4 → bf16. Verify against
   `compressed_tensors`' own dequant.
2. Benchmark it against the unfused PyTorch sequence — you should see the HBM
   round-trip cost directly.
3. *Then* fuse it into a GEMV (Level 4).

**Where it plugs into this repo:** our `compressed-tensors` checkpoints
(`outputs/qwen3-1.7b/awq-w4a16/`) in the HF runtime path
(`src/runtimes/hf/hf_runtime.py`). A fused path here is what would close the eager-vs-vLLM gap.

**Skills:** bit manipulation in Triton (`>> 4`, `& 0xF`), grouped scale indexing,
`tl.constexpr` for compile-time bit widths.

**Realistic outcome:** you will *not* beat Marlin (years of tuning). You will understand
exactly why Marlin is fast, which is the actual goal at this stage.

---

## Level 3 — TurboQuant Kernel A: MSE score ⭐ first repo-specific kernel (~1 week)

**What we're doing:** TurboQuant stores each cached key as packed codebook *indices*
plus a norm. To score a query against a key, the PyTorch reference rebuilds the full
key vector (gather centroids → multiply → materialize `k_hat`) and then dots it. The
kernel skips the materialization: stream packed indices, gather centroids into
registers, accumulate the dot product directly.

$$\text{score}_i = \|k_i\| \cdot \sum_{j=1}^{d} q_{\text{rot}}[j] \cdot \text{codebook}[\text{idx}_i[j]]$$

Full signature, pseudocode, and block-size guidance already written up in
[`notes/turboquant.md`](../notes/turboquant.md) §5.2 Kernel A — that doc is the spec,
this is the context for why it's the right first one.

**Why it's the right first repo kernel:** strictly streaming, no inter-block
communication, one tunable (`BLOCK_N`), and a dead-simple correctness check — the
PyTorch reference already exists in `src/turboquant_v1/score.py`.

**Where it plugs in:** `src/turboquant_v1/` — replaces the score path used during decode.

**Target (from `notes/kernel_scope.md`):** ~5x over the PyTorch baseline at N_k=8192.

---

## Level 4 — Packed W4A16 GEMV for batch=1 (~2 weeks)

**What we're doing:** the full fused operation — weight matrix stays packed in int4 the
entire time; dequant happens in registers inside the matmul loop; nothing intermediate
touches HBM. This is the Marlin-class kernel.

Harder than Level 2 because now you're managing: tiling the weight matrix, pipelining
loads against compute (load tile N+1 while computing tile N), and register pressure.

**Why batch=1 specifically:** Marlin is tuned for batch≥8. At batch=1 there's less
parallelism to hide latency, and it's the single-user chatbot case — the one our own
leaderboard measures. `notes/kernel_scope.md` §2.5 scopes this as a genuine (if
ambitious) opportunity: parity or +10–20% over Marlin at batch=1 on Ampere.

**Skills:** double-buffering, `tl.dot`, autotuning over tile shapes, reading PTX/ncu
profiles.

---

## Level 5 — Fused decode attention (TurboQuant Kernel C) ⭐ the real prize (~3-4 weeks)

**What we're doing:** one kernel that streams the KV cache exactly once and does
*everything* — score against compressed keys, online softmax, gather and dequantize
values, weighted-sum into the output — without ever materializing the attention matrix
or the dequantized K/V.

The core idea is **online softmax** (FlashAttention-2): normally softmax needs all
scores before you can normalize (two passes = two HBM reads). Online softmax keeps a
running max `m`, running sum `l`, and running output `o`, rescaling as each new block
arrives:

```
m_new = max(m_prev, block_scores.max())
p     = exp(block_scores - m_new)
l_new = exp(m_prev - m_new) * l_prev + p.sum()
o     = exp(m_prev - m_new) * o + (p @ block_values)
```

One pass, mathematically identical result. Full version with the TurboQuant-specific
value-side gather is in [`notes/turboquant.md`](../notes/turboquant.md) §5.2 Kernel C.

**Why it's the prize:** this is what converts TurboQuant's *memory* win (already
measured: 4x context) into a *speed* win. Current TQ decode is 4.3 TPS vs fp16's 21.5
because every step runs unfused. Target from `kernel_scope.md`: ≥1.2x FP16 at ctx=32K
while using ≥3x less KV VRAM.

**Known risk on our hardware** (already flagged in `kernel_scope.md` §6): register
pressure at D=128 on sm_86 may spill. Mitigation: split D across two warps, or process
keys in groups of 64.

---

## Level 6+ — Beyond this repo's current scope

Listed so you know they exist and roughly what they're for; `notes/kernel_scope.md`
covers the reasoning for deferring each.

| Topic | What it's for | Why deferred |
|---|---|---|
| TileLang / FlashQLA-style kernels | Cross-vendor (CUDA+ROCm), heavy mixed-dtype tiling | New DSL, ~2 weeks learning; Triton covers our NVIDIA-only needs |
| Raw CUDA + PTX | Last 10–20% Triton can't express; warp-level primitives | Only worth it after Triton plateaus |
| CUTLASS | Production GEMM template library | Steep; industry-standard for custom GEMM |
| NVFP4/MXFP4 dequant on Ampere | Would let us *test* NVFP4 without Blackwell | No Tensor Core support on sm_86 — correctness only, no speed |
| Custom collectives | Multi-GPU | Single-GPU box |

---

## How to validate anything you write (non-negotiable)

`notes/kernel_scope.md` §5 has the full methodology. The short version:

1. **Correctness first.** `torch.allclose(triton_out, torch_ref, atol=1e-3, rtol=1e-2)`
   on 100 random inputs; NaN sweep on 1000.
2. **Then benchmark.** `triton.testing.do_bench(fn, warmup=25, rep=100)` — median plus
   p20/p80. A speedup you measured once is not a speedup.
3. **Then roofline.** Compute arithmetic intensity; confirm you're memory-bound where
   you expect to be. If you think you're compute-bound at batch=1 decode, you've made
   an arithmetic error.
4. **Then end-to-end.** Token agreement ≥99% vs FP16 on greedy decode — a kernel that's
   fast and subtly wrong is worse than no kernel. This repo has already been bitten by
   silent-wrong twice (see [`docs/failure_cases.md`](failure_cases.md) #1 and #4); the
   debugging methodology in
   [`notes/debugging_turboquant_kv.md`](../notes/debugging_turboquant_kv.md) is directly
   applicable.

---

## Suggested first move

Level 0 → Level 1 → **Level 2 stopping at the standalone dequant benchmark**. That
single experiment reproduces the 13.6x gap from our leaderboard with code you wrote
yourself, and it's the cheapest path from "I read about memory-bound kernels" to "I
measured one." Everything after that is a matter of degree.
