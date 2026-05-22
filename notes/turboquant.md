# TurboQuant — From-Scratch Implementation & Kernel Optimization

**Scope:** weeks 2–3 of the 3-week sprint (see `plan.md`).
**Goal:** ship a working TurboQuant KV-cache quantizer for Llama-3-8B with Triton-fused decode attention, beating FP16 tok/s at ctx ≥ 8K.

**Source material:**
- TurboQuant paper: Zandieh, Daliri, Hadian, Mirrokni — *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (arXiv 2504.19874, ICLR 2026).
- PolarQuant paper: arXiv 2502.02617 (AISTATS 2026).
- QJL paper: Zandieh et al., *Quantized JL: Sub-Linear Quantization of KV Cache* (AAAI 2025).
- Reference impl: github.com/0xSero/turboquant (GPL-3.0 — do NOT vendor code; read for design only).

---

## 1. The Idea in One Paragraph

A transformer's attention only ever uses two operations on K, V: inner product `⟨q, k⟩` and weighted sum `Σ p_i v_i`. Standard quantization minimizes reconstruction MSE per vector — overkill. TurboQuant minimizes distortion of the *inner product*, which is what attention actually consumes. The trick: random-rotate every vector to a uniform distribution on the sphere, scalar-quantize each rotated coordinate with a Lloyd-Max codebook (cheap, no per-tensor scale, no outliers), then add a 1-bit per-coordinate sign correction (QJL) that makes the inner-product estimator *unbiased*. At 3.5 bits/channel this is quality-neutral to FP16; at 2.5 bits/channel degradation is marginal. KV-cache size: 4–6× smaller.

---

## 2. Why It Works (Math)

### 2.1 Random rotation flattens outliers

Let `x ∈ R^d` be a K or V vector (`d = head_dim`, typically 128 for Llama-family).
Pick a fixed random orthogonal matrix `Π ∈ R^{d×d}` (one per layer or shared globally; just must be invertible and norm-preserving).

Define `y = Π · (x / ‖x‖₂)`.

Since `Π` is orthogonal:
- `‖y‖₂ = 1` (still on the unit sphere).
- `⟨Πa, Πb⟩ = ⟨a, b⟩` (inner products preserved).

But the *coordinates* of `y` are now i.i.d.-ish: each `y_i` is approximately distributed as the marginal of a uniform point on `S^{d-1}`, which is a scaled Beta(½, (d-1)/2) → converges to `N(0, 1/d)` for large `d`. This is the key: **the per-coordinate distribution is known, fixed, and outlier-free.** No per-channel scale, no per-token scale.

### 2.2 Lloyd-Max codebook is optimal for this distribution

Since every coordinate has the same known distribution `f(y)`, build one global codebook with `2^b` centroids `{c_1, …, c_{2^b}}` that minimize `E_{y~f}[(y - Q(y))²]` via Lloyd-Max iteration. Apply to every coordinate of every vector everywhere. One codebook for the whole model.

Decision boundaries are the midpoints between consecutive centroids. Encoding = `searchsorted(boundaries, y)`. Decoding = `centroids[idx]`.

### 2.3 QJL: 1-bit residual correction for unbiased inner products

After MSE quantization, residual `r = x - x̂` carries the leftover info. Standard quant throws it away. TurboQuant compresses it to **1 bit per coordinate** via a Quantized Johnson-Lindenstrauss transform:

- Sample a random Gaussian matrix `S ∈ R^{d×d}` (fixed, shared, same seed as `Π` family).
- Project residual: `s = S · r ∈ R^d`.
- Store only `sign(s) ∈ {-1, +1}^d`, packed as 1 bit per coord.
- Store `‖r‖₂` (one fp16 scalar per vector).

To estimate `⟨q, x⟩` use:
```
⟨q, x̂_MSE⟩ + (√(π/2) / d) · ‖r‖₂ · ⟨S q, sign(S r)⟩
```

The constant `√(π/2)/d` makes the second term an **unbiased** estimator of `⟨q, r⟩` (proof: standard Gaussian sign-bit sketch). So the full estimator is unbiased for `⟨q, x⟩`.

### 2.4 Bit budget

For Llama-3 `head_dim = 128`:

| Scheme | MSE bits | QJL bits | Norm bits | Total/coord | KV size vs FP16 |
|---|---|---|---|---|---|
| Pure MSE 3-bit | 3 | 0 | 16/128 ≈ 0.125 | 3.125 | 5.1× |
| TurboQuant_Prod K=3 V=2 | (K:2, V:1) + 1 QJL | 1 each | 0.25 | 3.25 / 2.25 | ~5× |
| Paper "quality neutral" | 3.5 effective | bundled | tiny | 3.5 | 4.6× |
| Paper "marginal degradation" | 2.5 effective | bundled | tiny | 2.5 | 6.4× |

Keys are more sensitive than values (small perturbations move softmax probs a lot). Asymmetric: keys 3–4 bits, values 2–3 bits.

### 2.5 Caveat (verified by independent reimplementations)

Multiple open-source teams (`turboquant-mlx`, `turboquant-pytorch`, `0xSero/turboquant`) found **MSE-only outperforms MSE+QJL in practice on real LLM workloads** — the QJL term adds variance that hurts decode quality more than its unbiased-ness helps. Recommended default: implement *both*, ship MSE-only, leave QJL behind a `--use-qjl` flag.

---

## 3. Algorithm — Pseudocode (PyTorch reference)

### 3.1 Initialization (once at load time, per `head_dim`)

```python
Π = generate_rotation_matrix(d, seed=42)   # QR(N(0,I)), sign-fix diag → orthogonal
S = generate_qjl_matrix(d, seed=1042)      # N(0,1) entries, NOT orthogonalized
codebook = lloyd_max(pdf=beta_marginal(d), n_clusters=2**b, iters=200)
boundaries = midpoints(codebook)
```

### 3.2 Encode K (or V), MSE-only variant

```python
def encode_mse(x, bits):              # x: (..., d)
    n = x.norm(dim=-1, keepdim=True)  # (..., 1)
    u = x / (n + 1e-10)               # unit sphere
    y = u @ Π                         # rotate; (..., d)
    idx = searchsorted(boundaries[1:-1], y)   # (..., d) int in [0, 2^b)
    return pack(idx, bits), n.squeeze(-1)     # uint8 packed, fp16 norm
```

### 3.3 Decode (slow ref path — full materialization)

```python
def decode_mse(packed, norm, bits):
    idx = unpack(packed, bits, d)     # (..., d) long
    y_hat = codebook[idx]             # (..., d)
    u_hat = y_hat @ Π.T               # un-rotate
    return u_hat * norm.unsqueeze(-1) # (..., d)
```

### 3.4 Fast attention score (no materialization)

The whole point: never reconstruct `k_hat`. Compute `⟨q, k_hat⟩` directly:

```python
# Rotate query forward, once per decode step.
q_rot = q @ Π                                # (d,)

# For each cached key i:  score_i = ‖k_i‖ · Σ_j q_rot[j] · codebook[idx_i[j]]
# i.e. a per-coord centroid-gather then dot.
score = norm * (q_rot[None, :] * codebook[idx]).sum(dim=-1)
```

This is the kernel-friendly form. No `d`-sized intermediate per key — just a streaming gather + dot.

### 3.5 Prod variant (with QJL)

Encode adds a second stage:
```python
def encode_prod(x, mse_bits):
    mse_idx, norm = encode_mse(x, mse_bits)
    x_hat = decode_mse(mse_idx, norm, mse_bits)
    r = x - x_hat
    r_norm = r.norm(dim=-1)
    signs_packed = pack_signs(r @ S.T > 0)
    return mse_idx, signs_packed, norm, r_norm
```

Score adds a second term:
```python
q_sketch = q @ S.T                                  # (d,)
score_qjl = (q_sketch[None, :] * unpack_signs(signs)).sum(dim=-1)
score_qjl = score_qjl * (math.sqrt(math.pi / 2) / d) * r_norm
total_score = score_mse + score_qjl
```

### 3.6 Value side

Same MSE encode/decode. Weighted sum during attention:
```python
out = Σ_i p_i · decode_mse(v_packed_i, v_norm_i)    # (d,)
```
Or fused: `out[j] = Σ_i p_i · v_norm_i · codebook[v_idx_i, j]`, then `out @ Π_v.T`.

---

## 4. From-Scratch Implementation Plan (Week 2)

### 4.1 Directory layout

```
src/turboquant/
  __init__.py
  rotation.py         # generate_rotation_matrix, generate_qjl_matrix, rotate_fwd/bwd
  codebook.py         # Lloyd-Max + disk JSON cache (~/.cache/triplequant/codebooks/)
  pack.py             # pack_indices / unpack_indices / pack_signs / unpack_signs
  quantizer.py        # TurboQuantMSE, TurboQuantProd (nn.Module, register_buffer)
  kv_store.py         # CompressedKVStore — per-layer ring buffer of (idx, norm[, signs, r_norm])
  capture.py          # forward_hook on attention to intercept K/V, swap cache
  attention.py        # python reference decode-step attention over compressed store
  integration/
    __init__.py
    hf_llama.py       # monkey-patch LlamaAttention.forward when active
```

### 4.2 Build order (Day-by-day)

**Day 8 — rotation + codebook**

- `generate_rotation_matrix(d, seed)` — `Q, R = qr(randn(d, d)); Q *= sign(diag(R))` → uniform Haar orthogonal. Float32 buffer, 64KB for d=128, negligible.
- `generate_qjl_matrix(d, seed)` — plain `randn(d, d)`. No QR.
- `lloyd_max(d, bits)` — iterative k-means on samples from Beta(½, (d-1)/2). Initialize centroids at quantiles. Loop: assign → mean → check delta. ~200 iters. Cache `(d, bits) → (centroids, boundaries)` to `~/.cache/triplequant/codebooks/cb_d{d}_b{bits}.json`.
- Unit test: `assert (Π @ Π.T).allclose(I, atol=1e-5)`; `assert codebook.shape == (2**bits,)`; round-trip MSE on synthetic Gaussian within theoretical bound.

**Day 9 — bit packing**

- `pack_indices(idx, bits)` for bits ∈ {1,2,3,4}. bits=3 packed as 4-bit (waste 1 bit/coord, simpler & still 2× smaller than uint8). Use `(idx << shifts).sum(dim=-1)` trick.
- `pack_signs(bool_tensor)` — 8 bools per uint8 byte via `*powers` reduce.
- Unit tests: random round-trip every bit-width; verify byte-count matches `ceil(d * bits / 8)`.

**Day 10 — quantizer correctness**

- `TurboQuantMSE(dim, bits)` and `TurboQuantProd(dim, bits)` as `nn.Module`. `register_buffer` for `Π`, `S`, `centroids`, `boundaries`.
- Tests:
  - **Reconstruction MSE** (5000 random unit vectors, d=128, bits=3): TurboQuant < per-coord uniform scalar quantization by ≥1.5×.
  - **Inner-product unbiasedness** (Prod variant): over 10K (q, k) pairs, `mean(est - true) / σ < 0.1`.
  - **Distortion vs paper**: at 3.5 effective bits, `‖x - x̂‖² / ‖x‖² < 0.05` (paper Fig. 2 reference).

**Day 11 — KV capture**

- `CompressedKVStore(num_layers, num_kv_heads, head_dim, bits_k, bits_v)`:
  - Per layer: list-of-tensors growing by one row per decode step.
  - Storage: `idx: (T, num_kv_heads, packed_d) uint8`, `norm: (T, num_kv_heads) fp16`.
- `attach(model)` → registers `forward_pre_hook` on every `LlamaAttention`, replacing the `past_key_value` mechanism. Capture K, V after RoPE (look at HF Llama source — RoPE is in `apply_rotary_pos_emb` before the attention compute).

**Day 12 — Python reference attention**

- Replace `LlamaAttention.forward` decode-step path: dequant K and V from store, do standard `softmax(Q K^T / √d_k) V`, return output. Slow, fine.
- Run `model.generate(prompt, max_new_tokens=128)` on Llama-3-8B-Instruct. Compare to FP16 baseline: token-agreement on greedy ≥ 95% for 3-bit, ≥ 99% for 4-bit.

**Day 13 — eval gate**

- `eval/turboquant_ppl.py`: wikitext-2 sliding window, ctx 2048, report PPL.
- Gates (Llama-3-8B-Instruct):
  - K=4, V=4, MSE only: ΔPPL ≤ 0.1
  - K=3, V=2, MSE only: ΔPPL ≤ 0.3
  - K=3, V=2, MSE + QJL: report side-by-side
- Baselines: FP16, FP8 KV (via modelopt or vLLM).

**Day 14 — config + CLI**

- `TurboQuantConfig` in `schemas.py`: `bits_k`, `bits_v`, `use_qjl`, `seed`, `rotation_shared` (bool: one Π across layers or per-layer).
- `@register("turboquant")` quantizer — but note: this is *not* weight quant. It produces a model that loads as FP16 weights + activates KV compression at runtime. Wrap as: load FP16, attach hooks, save state.
- Demo CLI: `python quantize.py --config config/turboquant/llama3_8b_k3v2.yaml` → saves a marker file + state dict pointer + serialized `(Π, S, codebook)` to `output_dir`.
- Demo eval: `python tests/turboquant_demo.py --model output_dir --prompt "..."` → loads, attaches, generates.

---

## 5. Kernel Optimization (Week 3)

### 5.1 Why kernels at all

The Python ref is bandwidth-bound on the dequant + materialization steps. At ctx=8K, each decode step does:
- Load 8K × num_kv_heads × packed_d bytes from HBM
- Unpack to int8
- Gather centroids
- Multiply by `q_rot`
- Reduce per row
- Softmax + V-side gather + reduce

Done naively in PyTorch: 5–10× slower than FP16 baseline despite 4× less KV memory.

Goal: a single Triton kernel that **streams over the KV cache once**, computes scores + softmax + value-aggregation, returns `(d,)` output. Same shape as Flash-Decode / Flash-Attention v2 decode kernel.

### 5.2 Three kernels (build incrementally)

#### Kernel A — MSE score kernel

Inputs:
- `q_rot: (D,) fp16` — query already rotated by `Π` (do once outside).
- `idx_packed: (N_k, D_packed) uint8` — packed K indices.
- `norms: (N_k,) fp16`.
- `centroids: (2^b,) fp32`.
- `bits: int constexpr`.

Output: `scores: (N_k,) fp32`.

```python
@triton.jit
def mse_score_kernel(q_rot_ptr, idx_ptr, norms_ptr, centroids_ptr, out_ptr,
                     N_k, stride_idx_n, D: tl.constexpr, BITS: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N_k

    # Load query once into registers
    offs_d = tl.arange(0, D)
    q = tl.load(q_rot_ptr + offs_d)                 # (D,)

    # Iterate keys in this block
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d_start in range(0, D, 8 // BITS):
        # Load packed bytes, unpack to BITS-wide ints
        ...
        c = tl.load(centroids_ptr + idx)            # gather centroid
        acc += c * q[offs_d_chunk][None, :]
    # acc shape (BLOCK_N, D) — reduce
    acc = tl.sum(acc, axis=1)
    n = tl.load(norms_ptr + offs_n, mask=mask_n)
    tl.store(out_ptr + offs_n, acc * n, mask=mask_n)
```

Block size: tune `BLOCK_N ∈ {32, 64, 128}` for occupancy. D=128 fits in a single warp's register file.

#### Kernel B — QJL score kernel

Same structure, replace `centroids[idx]` gather with `sign_bit → ±1`, multiply by pre-sketched query `q_sketch = q @ S.T`. Output is `(N_k,)` to be added to MSE scores. Scaled by `sqrt(π/2)/d · r_norm`.

#### Kernel C — fused decode (the real win)

Inputs: K-side packed + V-side packed. Output: `(D,) fp16`.

Strategy (Flash-Attention v2 style):
1. Split keys into blocks of `BLOCK_N` (e.g., 64).
2. For each block:
   - Compute scores (kernel A logic, inlined).
   - Online softmax: track running `(max, sum, output)` per query.
   - Compute partial output by gathering V centroids by `idx_v`, weighting by softmax-normalized scores, accumulating into `out` register tile (D,).
3. After last block: divide `out / sum`, multiply by `Π_v.T` (or store rotated, derot at the end — cheaper if D=128).

Key trick: never write the full attention probabilities to HBM. Whole computation in SRAM/registers per query (decode = 1 query per call, easy).

Pseudo:
```
out = zeros(D)
m_prev = -inf; l_prev = 0
for block in chunks(KV):
    s = mse_score(q_rot, block_k_idx, block_k_norms, codebook)      # (BLOCK_N,)
    s += qjl_score(q_sketch, block_k_signs, block_k_rnorms)         # if use_qjl
    s = s / sqrt(D)
    m_new = max(m_prev, s.max())
    p = exp(s - m_new)
    l_new = exp(m_prev - m_new) * l_prev + p.sum()
    # Value-side gather + weighted sum
    v_partial = (p[:, None] * codebook_v[block_v_idx] * block_v_norms[:, None]).sum(axis=0)  # (D,)
    out = exp(m_prev - m_new) * out + v_partial
    m_prev, l_prev = m_new, l_new
out = out / l_prev
out = out @ Π_v.T
```

#### Kernel D (optional stretch) — fused prefill quantize

After prefill, K and V are `(T, num_kv_heads, D)` FP16. One pass: rotate, normalize, searchsorted, pack — all in one kernel. Saves a memory round-trip vs PyTorch sequence of ops.

### 5.3 Optimization knobs (in order of impact)

1. **Pre-rotate query once outside the kernel.** Don't rotate inside per-block.
2. **Reciprocal-trick for unpacking.** For 4-bit packing: `lo = byte & 0xF; hi = byte >> 4`. Triton handles `tl.where`/bitwise natively.
3. **codebook in shared memory / constant memory.** Only 8–16 floats; fits in registers. Pass as `tl.constexpr`-friendly load once.
4. **BLOCK_N tuning.** Sweep 32, 64, 128, 256. Profile with `triton.testing.do_bench`. Expect 64 or 128 wins on A100/H100.
5. **Π · q_rot precision.** Π is fp32 buffer; `q_rot` can stay fp16 — matmul in fp32 accumulator, cast back.
6. **vmap over `num_kv_heads`.** Launch grid `(ceil(N_k/BLOCK_N), num_kv_heads)`. Each program handles one head's slice — embarrassingly parallel.
7. **Avoid bank conflicts** on codebook gather: pad codebook size to 16 if `2^b < 16`.
8. **Persistent kernel** (if Triton supports on target) — one program per SM, loop over query tokens for batch decode. Marginal gain for batch=1.

### 5.4 Validation gates (Day 18)

Per kernel, before chaining:
- Bitwise-equivalent to PyTorch ref on 100 random inputs, `atol=1e-3, rtol=1e-2` (Triton fp32 reduce slightly differs from torch fp32 reduce).
- No NaNs across 1000 random inputs.
- Latency: kernel A alone < 50 µs for `N_k=8192, D=128` on A100.

### 5.5 End-to-end benchmark plan (Day 20)

Setup: Llama-3-8B-Instruct, batch=1, greedy, max_new_tokens=128.

| Config | Ctx 1K | Ctx 8K | Ctx 32K |
|---|---|---|---|
| FP16 (baseline) | tok/s, peak VRAM | … | … |
| FP8 KV (modelopt) | … | … | … |
| TurboQuant K=4/V=4 MSE | … | … | … |
| TurboQuant K=3/V=2 MSE | … | … | … |
| TurboQuant K=3/V=2 + QJL | … | … | … |

Goals:
- At ctx=32K: TurboQuant K=3/V=2 ≥ 1.2× FP16 tok/s, ≥ 3× less KV VRAM.
- At ctx=1K: parity with FP16 (kernel overhead dominates; expected).

### 5.6 Risks & mitigations

| Risk | Mitigation |
|---|---|
| Triton version mismatch with installed PyTorch | Day 15 smoke test first; pin version in `req.txt` |
| Per-head Π different on different layers — too many constants | Default: one global Π. Validate accuracy doesn't tank. |
| Searchsorted in Triton is awkward | Encode is offline (one-shot prefill); keep encode in PyTorch. Triton only handles *decode*. |
| Kernel A slower than torch on small N_k | Add fallback: if `N_k < 512`, use torch path. |
| HF generate path expects `past_key_value` tuple | Subclass and override `prepare_inputs_for_generation` to feed our `CompressedKVStore` as a custom cache object. |
| QJL hurts accuracy | Already planned: ship MSE-only by default, QJL behind flag. |
| Triton OOMs on Windows + 30-series | Develop on Linux/A100 if available; document Windows caveats. |

---

## 6. Acceptance Criteria (sprint-end demo)

- [ ] `quantize.py --config config/turboquant/llama3_8b_k3v2.yaml` runs to completion in < 5 min.
- [ ] `tests/turboquant_demo.py` produces coherent generation matching FP16 token-agreement ≥ 95%.
- [ ] wikitext PPL (ctx 2048): K=3/V=2 within +0.3 of FP16, MSE-only mode.
- [ ] Triton kernels A/B/C all pass numerical equivalence test.
- [ ] Benchmark table populated for ctx ∈ {1K, 8K, 32K}.
- [ ] One bench config wins on both VRAM and tok/s at ctx=32K vs FP16.
- [ ] README / `notes/turboquant.md` cross-link, and a 1-page result summary in repo root.

---

## 7. Open questions

1. **Π shared across layers, or per-layer?** Shared = one 64KB buffer + simpler codebook reuse, possible accuracy hit. Per-layer = `num_layers × 64KB` ≈ 2MB for Llama-3-8B (32 layers) — negligible, better accuracy. **Default per-layer, configurable.**
2. **Quantize K *before* or *after* RoPE?** Before = simpler hooks; after = what attention actually consumes. **Pick: after RoPE.** Required for correctness — rotation interacts with the rotary embedding.
3. **GQA grouping** — Llama-3-8B uses 8 KV heads for 32 Q heads. Quantize at KV-head level (8× smaller cache footprint already). Code handles this implicitly if we hook the K/V output, not Q.
4. **Eviction / sliding window?** Out of scope for v1. Document.
5. **TurboQuant on V vs only K?** Paper recommends both, asymmetric bits. We follow.
6. **vLLM integration** in week 3? Stretch only. HF `generate` is the primary deliverable.

---

## 8. References

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874 (ICLR 2026).
- *PolarQuant: Quantizing KV Caches with Polar Transformation.* arXiv:2502.02617 (AISTATS 2026).
- Zandieh et al. *QJL: 1-bit Quantized JL Transform for KV Cache.* AAAI 2025.
- Google Research blog — *TurboQuant: Redefining AI efficiency with extreme compression.*
- frr.dev — *TurboQuant one month later* (independent reproductions; MSE > MSE+QJL finding).
- github.com/0xSero/turboquant — reference impl (GPL-3.0, design-only reference; do not copy code).
- Dao. *FlashAttention-2.* arXiv:2307.08691 (decode-kernel structure).
