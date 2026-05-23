# Kernel Optimization Scope — Triton + TileLang

**Updated:** 2026-05-23
**Scope:** survey of kernel-level optimizations reachable inside the 3-week sprint (and beyond). Two languages in play: **Triton** (Python-embedded, mature, already in our `req.txt` via PyTorch) and **TileLang** (newer, tile-DSL, used by Qwen's FlashQLA — needs learning). Goal of this doc: enumerate every place in `TripleQuant-VLM` where a custom kernel buys real perf, classify each by language fit + difficulty + estimated speedup, and give a learning path for TileLang.

Companion docs: `notes/turboquant.md` (§5 already covers TurboQuant kernels A–D), `notes/benchmark.md` (perf eval harness).

---

## 0. TL;DR

| Priority | Kernel | Language | Difficulty | Est. speedup | Sprint week |
|---|---|---|---|---|---|
| P0 | TurboQuant fused decode (MSE score + softmax + V-aggregate) | Triton | Hard | 5–10× over Python ref; ≥1.2× over FP16 at ctx 32K | W3 |
| P0 | TurboQuant MSE score (Kernel A) | Triton | Med | ~5× over PyTorch | W3 (warmup) |
| P1 | TurboQuant QJL score (Kernel B) | Triton | Med | additive to A | W3 (optional) |
| P1 | Fused prefill quantize (rotate + norm + searchsorted + pack) | Triton | Med | 2× over PyTorch sequence | W3 stretch |
| P2 | FlashQLA-style fused QK-V for GQA decode (FP8/INT4 K/V) | TileLang | Hard | competitive with vLLM's AITER on MI300X; ~1.3× on Ampere | post-sprint |
| P2 | Packed W4A16 GEMV decode (small-batch latency) | Triton | Med | parity/beat Marlin at bs=1 | post-sprint |
| P3 | Lloyd-Max searchsorted prefill | Triton | Easy | 3–5× over `torch.searchsorted` on int8 | W3 stretch |
| P3 | Per-channel sign-bit GEMV (QJL inference side) | Triton | Easy | 4–8× over PyTorch | W3 optional |
| P4 | OCR VLM image-patch projection fuse | Triton | Med | 1.2–1.5× VLM prefill | post-sprint |
| P4 | NVFP4/MXFP4 dequant + GEMV (Ampere emulation) | TileLang | Very hard | enables NVFP4 on sm_86 at all | post-sprint research |
| P5 | LaTeX-tokenizer-aware CER (CPU, parallel) | not kernel — skip | — | — | — |

P0–P1 land **in-sprint** (week 3 has 6 dev days). P2+ are post-sprint candidates documented for the report.

---

## 1. Why FlashQLA matters (and what it teaches us)

Qwen team shipped **FlashQLA** (Flash QK-V Lookup-Aware) — a fused decode-attention kernel written in TileLang specifically for **GQA with low-bit KV cache** on Hopper/MI300X. Public release Mar 2026 alongside Qwen3.

### 1.1 What FlashQLA does

- Decode-only (single query per call; prefill stays FP).
- KV cache stored in low-bit format (FP8 E4M3 default; INT4/INT8 supported).
- Fuses: dequant K → `QK^T` → online softmax → dequant V → `softmax(...) · V` → output.
- Tile-level streaming over the KV sequence — never materializes the full attention matrix.
- GQA-aware: query head fans out within tile; KV head loaded once per tile, shared across the query-head group.
- TileLang lowers to PTX/AMDGCN with explicit shared-memory tiling and async copy.

### 1.2 Why TileLang (not Triton) for FlashQLA

Triton excels at single-stream GEMM-like kernels, but FlashQLA has:
- **Heterogeneous data types per tile** (FP8 K, FP8 V, BF16 Q, FP32 accumulator). Triton's type system handles this but verbosely.
- **Hardware-specific dequant intrinsics** (`cvt.rn.f16.e4m3` on Hopper; AMD's `v_cvt_pk_fp8_f32`). TileLang exposes these directly; Triton routes through LLVM intrinsics.
- **Cross-vendor portability**: TileLang has first-class CUDA + ROCm + CPU back-ends. Same kernel runs on H100 and MI300X with a config swap. Triton ROCm exists but trails CUDA features.
- **Tile-level autotune**: TileLang's autotuner explores tile shapes / pipeline depth / async-copy stages declaratively. Triton autotune is decorator-based and less expressive.

### 1.3 Takeaway for us

For **TurboQuant** kernels — single-tensor-type-mostly, NVIDIA-first — **Triton is the right tool** (also: we already have it installed, no learning curve, faster iteration in W3). For **cross-vendor fused KV-decode at production scale** — TileLang. We document the TileLang path as **post-sprint**, and use FlashQLA as a *design reference* for our Triton Kernel C (TurboQuant fused decode).

---

## 2. Kernel-by-kernel scope

### 2.1 TurboQuant Kernel A — MSE score [Triton, P0]

Already specified in `turboquant.md` §5.2 Kernel A. Recap:

- Input: `q_rot (d,)`, `idx_packed (N_k, packed_d) uint8`, `norms (N_k,) fp16`, `centroids (2^b,) fp32`.
- Output: `scores (N_k,) fp32`.
- Op: unpack indices → gather centroids → dot with rotated query → multiply by norm.

**Why it's a good first kernel:** strict-streaming, no inter-block sync, easy autotune (`BLOCK_N`).

**Speedup target:** 5× over PyTorch baseline at `N_k=8192, D=128`. Profiled with `triton.testing.do_bench`.

**Risks:** Triton's `searchsorted` is awkward — we sidestep by doing encode in PyTorch (offline) and only decode in Triton. `gather` on small codebook (8–16 floats) trivially fits in registers.

### 2.2 TurboQuant Kernel B — QJL score [Triton, P1]

- Input: `q_sketch (d,) fp16` (= `q @ S.T`, pre-computed once outside), `signs_packed (N_k, d/8) uint8`, `r_norms (N_k,) fp16`.
- Output: additive scores `(N_k,) fp32`.
- Op: unpack 8 signs/byte → `±1 · q_sketch[i]` per coord → reduce → scale by `√(π/2)/d · r_norm`.

Same skeleton as A. Cheap once A is working.

**Decision gate:** only ship if QJL improves PPL by >0.05 on eval. Independent reimplementations report MSE-only is better — we ship MSE-only by default (see `turboquant.md` §2.5).

### 2.3 TurboQuant Kernel C — Fused decode [Triton, P0, the headline]

The whole point. Single Triton kernel that:
1. Streams over KV tiles of `BLOCK_N`.
2. Per tile: MSE score (inlined A logic).
3. Online softmax — Flash-Attention v2 style `(m, l, o)` running stats.
4. V-side: gather V centroids by `idx_v`, multiply by softmax probs, accumulate into `o (D,)` register tile.
5. After last tile: divide by `l`, un-rotate by `Π_v.T`.

**Design reference:** FlashQLA tiling structure + FlashAttention-2 online-softmax math. We adapt to single-query (decode-only batch=1 v1, batch>1 stretch).

**Validation:** bitwise-equivalent to Python ref within `atol=1e-3` on 100 random inputs (Triton fp32 reduce diverges slightly from torch fp32 reduce — accept).

**Speedup target:**
- Ctx 8K: ≥0.9× FP16 baseline (kernel overhead amortizes).
- Ctx 32K: ≥1.2× FP16 baseline, ≥3× less KV VRAM.

**Risks:**
- Register pressure: `o (D=128,) fp32` + softmax state + tile-K state may spill on sm_86 (3060). Mitigation: split `D` across two warps; or process keys in groups of 64 instead of 128.
- Online softmax fp32 accumulation is mandatory; downcast only at end.
- HF `generate` path: subclass `Cache` (not the old `past_key_value` tuple) — current HF Llama uses `DynamicCache` / `StaticCache`. Override `update()` to call our packed store, override attention forward to call our kernel.

### 2.4 TurboQuant Kernel D — fused prefill quantize [Triton, P1 stretch]

After prefill, K/V is `(T, H_kv, D) fp16`. One pass:
- Rotate: `y = x @ Π / ‖x‖`.
- searchsorted on boundaries.
- Pack indices.
- Store `(idx_packed, norm)`.

Naive PyTorch does 5 separate ops, 5 HBM round-trips. Fused: one read, one write.

**Speedup target:** 2× over PyTorch op-sequence on T=8K. Memory-bound — wins come from reducing reads, not compute.

**Triton specifics:**
- `searchsorted` not native — implement as branchless bisect with `BITS ≤ 4` boundaries (≤16 compares, fully unrolled).
- Per-coord normalize is cheap: one reduce + one divide. Stays in registers.

**Decision gate:** ship only if prefill latency at T=8K exceeds 100ms with the unfused PyTorch path. Most users won't hit this; deprioritize accordingly.

### 2.5 Packed W4A16 GEMV — small-batch decode [Triton, P2 post-sprint]

vLLM uses Marlin for W4A16 — excellent at batch≥8, mediocre at batch=1 (fewer warps active, dequant overhead dominates).

A custom Triton kernel for batch=1 W4A16 GEMV could:
- Skip Marlin's batch-tile setup.
- Use vector-load for 4-bit packed weights, in-register dequant via shift + scale.
- Pipeline weight load with previous-row dot.

**Speedup target:** parity or +10–20% over Marlin at batch=1, D=4096+. Real impact: chatbot single-user TTFT.

**Why P2:** Marlin is *already* extremely tuned. Beating it on Ampere by ≥10% is realistic; beating on Hopper unlikely. Worth a few days of post-sprint work, not in-sprint.

### 2.6 LaTeX-aware searchsorted [Triton, P3]

Used in the TurboQuant encode path (offline). Currently `torch.searchsorted(boundaries, y)` — fine. A Triton kernel that fuses normalize + rotate + searchsorted on the *entire* prefill K/V at once would shave the encode pass. ~3–5× over the PyTorch sequence.

**Same kernel as 2.4.** They're the same problem; just covered twice in the plan if we ship D separately for V-side encode timing.

### 2.7 QJL inference-side GEMV [Triton, P3 optional]

If QJL ships: `q_sketch @ sign_bit_matrix` is a sign-times-vector reduce. 4–8× over PyTorch (which materializes `±1` floats first). Easy kernel: bit-test → conditional accumulate. ~30 LOC.

Only worth doing if QJL ships at all.

### 2.8 FlashQLA-style fused QK-V (GQA + low-bit KV) [TileLang, P2 post-sprint]

Same end-result as Kernel C, but:
- Uses TileLang tile primitives + autotune.
- Cross-vendor: same source compiles to PTX (Ampere/Hopper) and AMDGCN (MI300X).
- Properly GQA-aware (KV-head loaded once per query-head group inside the tile).
- Supports FP8/INT4/INT8 K and V via TileLang's typed-tile system.

**Why post-sprint:** TileLang is unfamiliar. Reference impl reading + reproduce-FlashQLA-on-Llama: ~2 weeks. Doable as a follow-up project, not within the 3-week sprint.

### 2.9 NVFP4 / MXFP4 dequant on Ampere [TileLang, P4 research]

NVFP4 has no native sm_86 kernel. vLLM emulates via Triton — slow + buggy. A TileLang kernel that dequants NVFP4 → fp16 in shared mem and feeds into a standard GEMM tile *could* make NVFP4 viable on 3060 for testing purposes.

**Reality check:** still no Tensor Core support for NVFP4 on Ampere. Best case: parity with FP16 perf, with 2× less VRAM. Not a perf win, but unlocks NVFP4 *correctness* testing without Blackwell. Niche but documentable.

### 2.10 VLM image-patch projection fuse [Triton, P4]

VLM prefill: patchify image → project → concat with text tokens → run LM. Patch projection is a small GEMM (e.g. `(N_patches, patch_dim) × (patch_dim, model_dim)`). Currently launches as a separate cuBLAS call.

Fuse into a Triton kernel that also handles RMSNorm + first transformer block's QKV proj? Maybe 1.2–1.5× VLM prefill. Niche; only matters if VLM prefill is on the hot path (it is for OCR pipelines).

---

## 3. TileLang — learning path

If we want post-sprint TileLang work, budget ~2 weeks of part-time learning + 1 week of FlashQLA-on-Llama port.

### 3.1 Resources (in order)

1. **TileLang repo** — `github.com/microsoft/BitBLAS` (TileLang lives here) → `tilelang/` subdir. Read `examples/` first, *not* the docs.
2. **TileLang quickstart** — official `docs/quick_start.md`. Tile primitive cheat sheet.
3. **FlashAttention-3 in TileLang** — `examples/flash_attention/` in the repo. Single best reference.
4. **FlashQLA source** — Qwen released as part of `qwen-attention-kernels` (Mar 2026). Read after FA3.
5. **TileLang autotune doc** — for kernel C-equivalent tuning.

### 3.2 Conceptual map (Triton → TileLang)

| Triton | TileLang | Notes |
|---|---|---|
| `tl.program_id(axis)` | `T.thread_binding(...)` / tile indices | Tile-level vs thread-level explicit |
| `tl.arange + tl.load` | `T.alloc_fragment + T.copy` | Explicit shared-mem fragments |
| `tl.dot` | `T.gemm` | Same idea, more dtype combos |
| `@triton.jit` | `@T.prim_func` + `T.kernel` | TVM-style schedule + kernel separation |
| autotune decorator | `T.autotune` with `T.Tunable` annotations | Declarative space |
| no async copy abstraction | `T.async_copy` + `T.commit_group` + `T.wait_group` | Maps directly to `cp.async` |

If you know FlashAttention-2's Triton implementation, you can read FlashAttention-3 in TileLang in a day.

### 3.3 Smallest learnable kernel (TileLang Hello-world)

Vector add → matmul → flash-attention-decode. Three kernels, ~2 days. Document the diff vs Triton for our future reference.

### 3.4 Decision tree

```
need new kernel?
  ├─ NVIDIA-only, single-tile-type, single-stream → Triton
  ├─ cross-vendor (CUDA + ROCm) → TileLang
  ├─ heavy mixed-dtype tiling (FP8 K, FP8 V, BF16 Q) → TileLang
  ├─ need to reference FlashQLA design → TileLang
  └─ else, prototype in Triton, port later
```

In-sprint: stay in Triton. TileLang work happens after the sprint demo lands.

---

## 4. Sprint-aligned kernel build order

Mapping to `notes/plan.md` Week 3 days (Day 15–21):

| Day | Kernel | Lang | Status |
|---|---|---|---|
| Day 15 | Triton smoke test (vector-add, GEMV) | Triton | env validation |
| Day 16 | Kernel A (MSE score) | Triton | P0 |
| Day 17 | Kernel B (QJL score) | Triton | P1 — skip if MSE-only ships |
| Day 18 | Kernel C (fused decode) | Triton | P0 — headline |
| Day 19 | HF `generate` integration; subclass `Cache` | Python | P0 |
| Day 20 | Benchmark sweep ctx 1K/8K/32K vs FP16 + vLLM-FP8-KV | Python | P0 |
| Day 21 | Buffer / writeup. Kernel D (prefill quantize) **only if Day 20 has spare** | Triton | P1 stretch |

Post-sprint queue (`v2 scope`): TileLang FlashQLA-on-Llama port, Marlin-rival W4A16 GEMV, NVFP4-on-Ampere dequant kernel, VLM patch-projection fuse.

---

## 5. Profiling & validation methodology

Every kernel goes through:

1. **Numerical equivalence test.** Random inputs N=100; assert `torch.allclose(triton_out, torch_ref_out, atol=1e-3, rtol=1e-2)`. NaN check N=1000.
2. **Microbench.** `triton.testing.do_bench(fn, warmup=25, rep=100)` → reports median µs + p20/p80. Compare to PyTorch baseline.
3. **Roofline.** Compute arithmetic intensity (FLOPs / bytes read+written). Plot vs achieved TFLOPs on `triton.testing` output. Confirm memory-bound vs compute-bound matches expectation.
4. **End-to-end gen.** Run `model.generate(prompt, max_new_tokens=128)` with kernel active. Token-agreement ≥99% vs FP16 baseline on greedy decode.
5. **Long-ctx stress.** ctx=32K, generate 256 tokens. No NaN/Inf in output. Peak VRAM logged.
6. **OCR sanity** (VLM models): top-50 worst CER samples on FP16 vs quantized kernel — visually inspect via Langfuse trace.

Tracked in W&B as `kernel/{name}/p50_us`, `kernel/{name}/tflops`, `kernel/{name}/parity_match_pct`.

---

## 6. Risks specific to kernel work

| Risk | Mitigation |
|---|---|
| Triton version drift breaks autotune | pin in `req.txt`; Day 15 smoke test |
| Windows + 30-series Triton flaky | develop kernels in WSL2 or Linux VM if breakage; document in README |
| sm_86 register spill on Kernel C | split tile D dim across two warps; benchmark both |
| HF `Cache` API change in transformers >4.45 | pin transformers version; subclass with both old/new API signatures |
| FlashQLA design assumes Hopper async copy | for our Ampere target, fall back to synchronous loads in Kernel C; less throughput but correct |
| TileLang build break on Windows | mark all TileLang work as Linux-only; document |
| Numerical divergence Triton vs torch in fp32 reduce | accept `atol=1e-3, rtol=1e-2`; flag if PPL/CER drift > 0.1 |

---

## 7. Acceptance criteria (kernel slice of sprint demo)

- [ ] Kernel A (MSE score) passes numerical equivalence + ≥5× over PyTorch.
- [ ] Kernel C (fused decode) passes numerical equivalence.
- [ ] End-to-end Llama-3-8B `generate` with TurboQuant K=3/V=2 produces coherent output (token-agreement ≥95% vs FP16).
- [ ] Benchmark sweep populated for ctx ∈ {1K, 8K, 32K}.
- [ ] At ctx=32K: TurboQuant K=3/V=2 ≥1.2× FP16 tok/s AND ≥3× less KV VRAM.
- [ ] Kernel B + D documented as "shipped/skipped" with rationale.
- [ ] `notes/kernel_scope.md` updated with measured numbers replacing "Est. speedup" in §0 table.
- [ ] One paragraph in README pointing to this doc + result chart.

---

## 8. References

- **FlashQLA** — Qwen team, Mar 2026. `github.com/QwenLM/qwen-attention-kernels`. TileLang impl of fused QK-V for GQA + low-bit KV.
- **TileLang** — Microsoft Research, lives inside `github.com/microsoft/BitBLAS/tree/main/tilelang`.
- **FlashAttention-2** — Dao, *FlashAttention-2*, arXiv:2307.08691. Online-softmax + tile-level streaming reference.
- **FlashAttention-3** — Shah et al., arXiv:2407.08608. Async-copy + WGMMA on Hopper. TileLang version in the BitBLAS examples.
- **Marlin** — `github.com/IST-DASLab/marlin`. W4A16 GEMM reference; what we'd compete against in §2.5.
- **Triton tutorials** — `triton-lang.org/main/getting-started/tutorials/`. Cover GEMM, softmax, flash-attention decode.
- **vLLM PagedAttention kernel** — `vllm/csrc/attention/`. Reference for production decode-attention shape + layout.
- **TurboQuant kernels** — `notes/turboquant.md` §5 (this repo). Math + Triton pseudocode for A/B/C/D.
- **HF `Cache` API** — `transformers/src/transformers/cache_utils.py`. The interface our Kernel C plugs into via subclass.
