# TurboQuant KV-cache in HuggingFace — how `hf_cache.py` works

A walkthrough of `src/turboquant_v1/tests/hf_cache.py`: how a TurboQuant compressed
KV cache is plugged into a stock HuggingFace model (Qwen2.5-3B), what every piece does,
why each transformers-compatibility hack is needed, and how to read the compression
numbers. Written as a study guide — read top to bottom once, then use it as a map.

---

## 0. The one-sentence idea

A transformer's KV cache stores the keys and values of every past token so each new
token doesn't re-attend from scratch. It grows linearly with sequence length and
dominates memory at long context. TurboQuant replaces that cache with a two-tier store:
the most recent N tokens kept exact (a "ring buffer"), older tokens quantized to ~2-3
bits. The script swaps Qwen2.5's cache for this store and intercepts attention so the
model reads from it.

There are two things to keep separate the entire time:
1. **Storage** — how many bytes the KV occupies. TurboQuant always shrinks this.
2. **Compute path** — what tensors attention actually multiplies. Whether TQ is "in the
   loop" depends on whether attention reads the *compressed* tokens or only the *exact*
   ring. This distinction is the crux (see §7).

---

## 1. Background you need first

### 1.1 KV cache
For each layer, attention projects the input into queries Q, keys K, values V. During
generation, K and V for all past tokens are cached so step *t* only computes Q/K/V for
the single new token, then attends Q against all cached K, weighting all cached V.
Cache size per layer = `seq_len x num_kv_heads x head_dim x 2 (K and V) x dtype_bytes`.
At bf16 that's 2 bytes/element. For Qwen2.5-3B: 36 layers, 2 KV heads, head_dim 128.

### 1.2 RoPE is baked into K before caching
Rotary position embeddings rotate Q and K by an angle that depends on token position.
Crucially, the rotation is applied **before** K goes into the cache. So cached keys
already carry their position. That's why the cache can be reordered/concatenated without
re-applying positions, and why our store can hold post-RoPE keys directly.

### 1.3 GQA (grouped-query attention)
Qwen2.5 has more query heads than KV heads (here 16 Q heads share 2 KV heads, ratio 8).
KV heads are repeated to match Q heads at attention time (`repeat_kv`). Our patch does the
same expansion with `repeat_interleave`.

### 1.4 Quantization bit budget -> compression
Keys quantized to 3 bits, values to 2 bits, vs 16-bit bf16. Ideal ratio is
`16 / ((3+2)/2) = 16 / 2.5 = 6.4x`. Real ratio is lower (~5.1x measured) because each
group also stores fp16 side-data: per-group scales/zeros for values, residual norms and
MSE norms for keys, and QJL sign bytes. Those overheads don't shrink with the bit budget.

---

## 2. How HuggingFace drives a cache during `generate`

`model.generate(..., past_key_values=cache)` runs, per step:

1. `model.forward` builds the causal mask. In transformers >=4.55 this calls
   `cache.get_mask_sizes(cache_position, layer_idx)` to learn `(kv_length, kv_offset)`.
2. Each decoder layer calls `self_attn.forward(hidden_states, position_embeddings,
   attention_mask, past_key_value=cache, cache_position=...)`. Note: **everything is
   passed by keyword**, `position_embeddings` is the precomputed `(cos, sin)`, and there
   is no `use_cache`/`position_ids` argument here anymore.
3. Standard attention would call `cache.update(k, v, layer_idx)` to append and get back
   the full K/V, then attend.
4. Generation also probes `cache.is_compileable` to decide whether to `torch.compile`.

A custom cache must satisfy the methods generation touches. That's the whole reason for
the compatibility methods in §4.

---

## 3. The two-tier store (the TurboQuant side)

Files: `src/turboquant_v1/capture.py`, `store.py`, `quantize.py`, `kv_cache.py`.

```
ingest_decode(k, v)              # one decode token per layer
        |
   RingBuffer.write              # capacity = RING_CAPACITY (e.g. 128)
        |  (returns overflow when full)
        v
CompressedKVStore.append_chunk   # quantizes overflow and stores it
        |
   TurboQuant.quantize (keys, K bits)  +  quantize_V (values, V bits)
```

- **RingBuffer** (`capture.py`): a fixed `capacity x num_kv_heads x head_dim` bf16 buffer
  for K and V. `write()` appends; when it fills, the oldest contents are returned as
  "overflow". `peek()` returns current contents without draining.
- **KVCaptureEngine** (`capture.py`): owns one ring + one store per layer. `ingest_decode`
  writes to the ring and, on overflow, calls `store.append_chunk` on the drained tokens.
- **CompressedKVStore** (`store.py`): keeps quantized chunks in lists (lazy concat),
  exposes `get_flat_cache()` (flattened quantized view), `memory_bytes()` (packed-size
  estimate), and the inverse via the quantizers.
- **Key quantizer** `TurboQuant` (`quantize.py`): two-stage — MSE codebook at `(bits-1)`
  bits, then a 1-bit QJL sign code on the residual. `quantize()`/`dequantize()` round-trip.
- **Value quantizer** `quantize_V`/`dequantize_V` (`kv_cache.py`): symmetric group
  quantization, bit-packed (4 x 2-bit per byte at V=2).

Net: the ring is exact recent context; the store is the quantized long tail.

---

## 4. The custom cache: `TurboDynamicCache`

Subclasses `transformers.cache_utils.Cache` only for `isinstance` compatibility — it does
**not** use the base layered machinery. It holds one `KVCaptureEngine` per layer and a
running `_seq_len`. Method by method:

- `update(key_states, value_states, layer_idx)` — HF hands `(1, Hkv, new_len, D)`. We drop
  the batch dim, permute to `(new_len, Hkv, D)`, and `ingest_decode` into that layer's
  engine. `_seq_len` is incremented once (on layer 0) per step.
- `get_full_kv(layer_idx)` — returns the full `(T, Hkv, D)` K and V for attention. This is
  where the ring (and optionally the decompressed store) is assembled. See §7.
- `get_seq_length()` — current total tokens; used by generation bookkeeping.
- `get_mask_sizes(cache_position, layer_idx)` — returns `(kv_length, kv_offset)` for mask
  construction. The base reads `self.layers[idx]` (which we don't have), so we compute it:
  `kv_length = _seq_len + query_length`, `kv_offset = 0`. (Our patched attention ignores
  the mask anyway — see §5 — but `model.forward` still builds it.)
- `__getitem__`, `to_legacy_cache`, `get_max_length`, `reorder_cache` — shape-probe and
  legacy shims generation occasionally calls.
- `is_compileable = False` (class attribute) — the base property iterates `self.layers`;
  this cache isn't `torch.compile`-able, so pin it False.

---

## 5. The attention monkeypatch

`Qwen2Attention.forward` is replaced globally by `patched_qwen2_attn_forward`. The patch
must match the transformers >=4.55 signature exactly:

```
forward(self, hidden_states, position_embeddings, attention_mask,
        past_key_value=None, cache_position=None, **kwargs)
```

Logic:
1. If `past_key_value` is not our cache, defer to the original forward (unchanged models
   keep working).
2. Project Q, K, V; reshape to `(b, heads, q_len, D)`.
3. Apply RoPE using the **passed** `(cos, sin)` from `position_embeddings`. Do not recompute
   from `position_ids` — 4.55 no longer passes it here; recomputing with `None` gives wrong
   positions and garbage output.
4. `past_key_value.update(k, v, layer_idx)` — append post-RoPE K/V to the cache.
5. `full_k, full_v = past_key_value.get_full_kv(layer_idx)` — read back all cached K/V.
6. GQA expand KV heads to Q heads; permute to `(1, Hq, T, D)`.
7. `F.scaled_dot_product_attention(..., is_causal=...)`. The `is_causal` flag is subtle:
   - **Prefill** (`q_len == kv_len > 1`): `is_causal=True` is correct (lower-triangular).
   - **Decode** (`q_len == 1`, `kv_len == T`): `is_causal=True` would mask the single query
     to key 0 only. Must be `False` so it attends the whole history.
   The patch sets `is_causal = (q_len == full_len and q_len > 1)`.

---

## 6. End-to-end data flow

**Prefill** (prompt, length P, with P < RING_CAPACITY):
- One forward over P tokens. Patched attention projects all P, RoPE, `update` writes P
  tokens to the ring (no overflow yet), attends with `is_causal=True`.

**Decode** (one token at a time):
- `update` appends 1 token to the ring. Once the ring exceeds RING_CAPACITY, the oldest
  block overflows into the store and is quantized.
- `get_full_kv` assembles K/V for attention, `is_causal=False`, single query attends all.

So as generation proceeds, the ring stays at RING_CAPACITY exact tokens and the store
accumulates the quantized older tokens.

---

## 7. The crux: is TurboQuant actually in the compute path?

`get_full_kv` decides this, via `USE_COMPRESSED_STORE`:

- **False (or the original TODO version)** — returns `ring.peek()` only. Attention sees at
  most RING_CAPACITY exact bf16 tokens: a **sliding window**. The store still fills (storage
  compression is real and measurable), but the model never reads compressed values. Output
  is numerically identical to a plain bf16 windowed cache. TQ is *not* in the loop.
- **True** — decompress the store and prepend it:
  ```
  flat   = store.get_flat_cache()
  store_k = store.quantizer.dequantize(flat.prod_q)        # (Hkv, T, D)
  store_v = dequantize_V(flat.value_q, store.value_group_size)
  full    = cat([store(reconstructed), ring(exact)], dim=tokens)
  ```
  Now attention runs over **lossy TQ-reconstructed** keys/values for the long tail. TQ is
  genuinely in the decode loop, and its reconstruction error affects generation.

This toggle is how to prove TQ is being used: flip it and the output changes. If it's
False, a clean answer tells you nothing about TQ — it's just the exact ring.

Order matters: the store holds older tokens, the ring newer, so concatenate
`[store, ring]`. RoPE is already baked into K, so no re-positioning is needed.

---

## 8. Measuring compression (`compression_report`)

After generation, for each layer:
- ring tokens stay bf16: `ring_tokens x Hkv x D x 2 bytes x 2 (K+V)`.
- store packed size: `store.memory_bytes()` (sums the packed index/sign/scale/zero tensors).
- "without TQ" reference = dense bf16 KV for the whole sequence:
  `seq_len x Hkv x D x 2 x 2 x num_layers`.

Reported ratios:
- **overall** = dense_fp16 / (ring_bf16 + store_packed). Dragged down by the bf16 ring;
  approaches the segment ratio as seq_len grows and the ring becomes a small fraction.
- **compressed-segment only** = fp16 cost of the stored tokens / their packed size. This is
  the true TurboQuant ratio (≈5.1x at K3/V2; ideal 6.4x, gap = fp16 side-data).

Observed at seq 520, ring 128: overall 4.81x, segment 5.12x (18.00 -> 3.52 MB).
To force the store to fill for measurement, `MIN_NEW_TOKENS` pushes generation past the
ring capacity (a short answer would never overflow and would report 1.00x).

---

## 9. transformers 4.55 compatibility fixes (why each exists)

The script was written for a pre-4.55 transformers; 4.55 refactored Cache + attention.
The fixes, in the order they were hit:

1. **`ModuleNotFoundError: No module named 'src'`** — running a file directly puts the
   file's dir on `sys.path`, not the repo root. Fix: insert `parents[3]` (the project root)
   on `sys.path` before importing `src...`.
2. **`Cache.__init__() missing 'layer_classes'`** — base init signature changed. The cache
   overrides everything it needs, so wrap `super().__init__()` in `try/except TypeError`.
3. **`'TurboDynamicCache' has no attribute 'layers'` in `is_compileable`** — base property
   iterates `self.layers`. Pin `is_compileable = False`.
4. **same in `get_mask_sizes`** — override it to compute `(kv_length, kv_offset)` directly.
5. **Garbage output** — two bugs: (a) the old patch signature used `position_ids`/`use_cache`,
   which 4.55 never passes, so the TQ branch never activated and the call fell through to a
   broken path; (b) `is_causal=True` at decode masks the query to key 0. Fixed the signature
   to use `position_embeddings` and the `is_causal` rule in §5.

---

## 10. Why the TQ-in-loop output degrades, and what to try

With `USE_COMPRESSED_STORE=True`, output stays coherent while answering, then collapses
into repetition (`to to to...`). That is the K3/V2 reconstruction error compounding: once
the exact ring tokens age out and attention depends on 3-bit keys / 2-bit values, attention
scores drift and decoding degenerates. It is *evidence TQ is in the loop* (a passthrough
cache cannot produce that), and a signal the bit budget is too low for this 3B model.

Next experiments:
- Raise bits: `KEY_BITS=4, VALUE_BITS=4` (compression ~3x, quality should hold).
- Isolate cause: compare `dequantize(quantize(x))` to `x` (per-layer MSE) to tell whether
  it's the bit budget or a reconstruction/index-packing bug.
- Larger ring: keep more exact recent tokens (helps short/medium context, less so long).
- Eventually: a fused Triton kernel that scores against compressed keys without
  materializing the dense K (the `notes/turboquant.md` / `kernel_scope.md` plan).

---

## 11. How to run

```bash
python src/turboquant_v1/tests/hf_cache.py        # from repo root
# or, equivalently:
python -m src.turboquant_v1.tests.hf_cache
```

Knobs at the top of the file: `MODEL_NAME`, `RING_CAPACITY`, `KEY_BITS`, `VALUE_BITS`,
`MAX_NEW_TOKENS`, `MIN_NEW_TOKENS`, `USE_COMPRESSED_STORE`, `PROMPT`. First run downloads
Qwen2.5-3B (~6 GB). The compression report prints after the generated text.

Related design docs: `notes/turboquant.md` (algorithm + kernel plan),
`notes/kernel_scope.md` (Triton scope).
