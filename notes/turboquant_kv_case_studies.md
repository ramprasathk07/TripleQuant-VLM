# TurboQuant KV-cache: bug-by-bug study guide

Companion to `notes/debugging_turboquant_kv.md` (the methodology) and
`notes/turboquant_hf_cache_guide.md` (how the code works). This file is case-based: each
bug we hit in `src/turboquant_v1/tests/hf_cache.py` gets a card with five parts —

- **Symptom**: the exact error or wrong behavior.
- **Why it happened**: the underlying mechanism (the part to actually learn).
- **How it was found**: the debugging step.
- **The fix**: before -> after code.
- **Lesson**: the transferable rule.

Background in one line: the script swaps Qwen2.5-3B's KV cache for a custom TurboQuant
cache and monkeypatches attention to read from it. It was written against a pre-4.55
transformers; transformers 4.55 refactored the `Cache` and attention APIs underneath it.
That mismatch is the source of cases 1–6.

---

## Case 1 — `ModuleNotFoundError: No module named 'src'`

**Symptom**
```
from src.turboquant_v1.store import CompressedKVStore
ModuleNotFoundError: No module named 'src'
```

**Why it happened**
Running `python src/turboquant_v1/tests/hf_cache.py` makes Python put the *script's own
folder* (`src/turboquant_v1/tests/`) on `sys.path[0]`, not the repository root. So the
absolute import `src.turboquant_v1...` can't be resolved — there is no `src/` under the
tests folder. The import statement is correct; the search path is wrong.

**How it was found**
The clue is in the failure mode itself: the import works under `python -m ...` but not as a
direct script. That pattern always points to `sys.path`, never to the module's contents.

**The fix** (add a path bootstrap before first-party imports)
```python
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # tests->turboquant_v1->src->ROOT
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ...then: from src.turboquant_v1.store import CompressedKVStore
```

**Lesson**
Import errors are environment/path problems, not code problems. Before touching the import,
ask "what is on `sys.path` at runtime?" Fix the path, or run with `-m`.

---

## Case 2 — `Cache.__init__() missing 1 required positional argument: 'layer_classes'`

**Symptom**
```
super().__init__()
TypeError: Cache.__init__() missing 1 required positional argument: 'layer_classes'
```

**Why it happened**
transformers 4.55 refactored `Cache` into a *layered* design: the base `__init__` now
requires a `layer_classes` argument describing per-layer cache types. Our custom cache
predates that and calls a no-arg `super().__init__()`. It also doesn't use the base layered
machinery at all — it manages its own per-layer engines and overrides every method
generation calls — so it doesn't actually need the base init.

**How it was found**
Bottom of the traceback points at the exact line (`super().__init__()`). The argument name
`layer_classes` tells you the base signature changed.

**The fix** (skip a base init this subclass doesn't rely on)
```python
try:
    super().__init__()
except TypeError:
    # transformers >=4.55 requires layer_classes; this cache overrides all access
    # points and doesn't use the base layered machinery, so skip base init.
    pass
```

**Lesson**
When you subclass a framework class but override its whole surface, a base `__init__` that
changed signature can be safely guarded — but only after confirming you don't depend on
what it sets up. Don't blindly pass dummy args you don't understand.

---

## Case 3 — `'TurboDynamicCache' object has no attribute 'layers'` (in `is_compileable`)

**Symptom**
```
transformers/cache_utils.py line 1286, in is_compileable
    return all(layer.is_compileable for layer in self.layers)
AttributeError: 'TurboDynamicCache' object has no attribute 'layers'
```

**Why it happened**
During `generate`, transformers checks `past_key_values.is_compileable` to decide whether to
`torch.compile` the decode step. The base `is_compileable` is a property that iterates
`self.layers` — a list the layered base init would have created. Our cache skipped that
(Case 2) and has no `layers`.

**How it was found**
Read the failing frame in the library source (`cache_utils.py:1286`). It iterates
`self.layers`; our cache has engines, not layers.

**The fix** (declare it directly; don't fake `layers`)
```python
class TurboDynamicCache(Cache):
    is_compileable = False   # class attribute shadows the base property
```

**Why not `self.layers = []`?** Then `all(... for ... in [])` returns `True` (vacuous
truth) — transformers would try to compile a cache that can't be compiled. The honest
answer is `False`.

**Lesson**
When stubbing to satisfy a framework probe, return the *semantically correct* value, and
watch empty-collection edge cases (an empty list makes `all()` True and `any()` False).

---

## Case 4 — same `no attribute 'layers'`, now in `get_mask_sizes`

**Symptom**
```
transformers/cache_utils.py line 1219, in get_mask_sizes
    kv_length, kv_offset = self.layers[layer_idx].get_mask_sizes(cache_position)
AttributeError: 'TurboDynamicCache' object has no attribute 'layers'
```

**Why it happened**
Before running the decoder layers, `model.forward` builds the causal mask and asks the cache
for `(kv_length, kv_offset)` via `get_mask_sizes`. The base delegates to
`self.layers[idx]` — again, not present.

**How it was found**
Same fix-rerun loop: the previous fix advanced execution to the *next* method the framework
calls on the cache. Read that frame, override it.

**The fix** (compute the sizes from our own state)
```python
def get_mask_sizes(self, cache_position, layer_idx=0):
    # called before decoder layers run: _seq_len is the PAST length, the current
    # query tokens (cache_position) are added on top. kv_offset = 0 (no sink).
    query_length = cache_position.shape[0]
    return self._seq_len + query_length, 0
```
(Our patched attention ignores the mask — it uses `is_causal` SDPA, Case 6 — but
`model.forward` still constructs it, so the call must not crash.)

**Lesson**
Framework subclass errors arrive one at a time; each fix reveals the next required method.
That's normal — treat the exception stream as the API contract checklist.

---

## Case 5 — garbage output `atsatsats...` (attention monkeypatch signature drift)

**Symptom**
No exception. Generation produced repeated nonsense like `atsatsatsats`.

**Why it happened**
The patch replaced `Qwen2Attention.forward`. transformers 4.55's real signature is
```
forward(self, hidden_states, position_embeddings, attention_mask,
        past_key_value=None, cache_position=None, **kwargs)
```
The old patch declared `(self, hidden_states, position_ids=None, past_key_value=None,
output_attentions=False, use_cache=False, cache_position=None, **kwargs)` and the decoder
passes everything by keyword. Result: `use_cache` was never passed (so the TQ branch,
gated on `if ... and use_cache`, never ran — it fell through to a path that misused the
cache), and RoPE was recomputed from `position_ids=None` (wrong positions). Two failures
stacked, both from the signature no longer matching the host.

**How it was found**
Printed the live signature:
```python
import inspect
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
print(inspect.signature(Qwen2Attention.forward))
```
That showed `position_embeddings` (not `position_ids`) and no `use_cache`, explaining both
the dead branch and the wrong RoPE.

**The fix** (match the signature; gate on cache type; use the passed cos/sin)
```python
def patched_qwen2_attn_forward(self, hidden_states, position_embeddings=None,
                               attention_mask=None, past_key_value=None,
                               cache_position=None, **kwargs):
    if not isinstance(past_key_value, TurboDynamicCache):
        return original_qwen2_attn_forward(self, hidden_states, position_embeddings,
            attention_mask, past_key_value=past_key_value,
            cache_position=cache_position, **kwargs)
    ...
    cos, sin = position_embeddings          # use what the model already built
    q, k = apply_rotary_pos_emb(q, k, cos, sin)   # do NOT recompute from position_ids
```

**Lesson**
A monkeypatch is only valid against one signature. Pin it with `inspect.signature` against
the *installed* framework. Gate on something the framework still passes (the cache type),
not on an argument that may have been removed (`use_cache`).

---

## Case 6 — still degraded: `is_causal=True` at decode

**Symptom**
After Case 5, prefill looked right but multi-step decode was still wrong.

**Why it happened**
The patch called `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. With
`is_causal=True`, SDPA builds a lower-triangular mask aligned top-left over a `q_len x
kv_len` grid. At decode `q_len == 1` and `kv_len == T`: the single query (row 0) is allowed
to see only key 0 — it cannot attend to the history at all. Correct for prefill
(`q_len == kv_len`), catastrophically wrong for decode.

**How it was found**
Pure reasoning about what the causal mask means at `q_len=1` — this never raises, so no
traceback. Thinking through the mask geometry exposed it.

**The fix** (causal only when query and key lengths match)
```python
is_causal = q_len == full_k.shape[2] and q_len > 1   # True at prefill, False at decode
attn_output = F.scaled_dot_product_attention(q, full_k, full_v, is_causal=is_causal)
```
After this, output was coherent: "The capital of India is New Delhi."

**Lesson**
Shape/masking bugs are silent. When `q_len != kv_len` (any cache scenario), reason
explicitly about whether `is_causal` does what you think; for single-token decode against a
long cache, you almost always want `is_causal=False`.

---

## Case 7 — "is it actually TurboQuant, or the baseline cache?"

**Symptom**
Coherent output — but was the model computing on quantized data, or just exact KV?

**Why the doubt was correct**
`get_full_kv` returned only `engine.ring.peek()` (the exact bf16 ring) with a literal
`# TODO: prepend compressed tokens from engine.store.decompress(...)`. The compressed store
*was* being written on ring overflow (so storage compression is real), but attention never
read it. So generation ran on an exact bf16 *sliding window* — numerically identical to a
plain cache. TQ was not in the compute path.

**How it was found / proven**
Read `get_full_kv` (found the TODO), then proved causality with an A/B toggle:
```python
USE_COMPRESSED_STORE = True   # flip to False -> ring-only
```
and wired the True branch to decompress the store and prepend it:
```python
flat   = engine.store.get_flat_cache()
store_k = engine.store.quantizer.dequantize(flat.prod_q).permute(1,0,2).to(ring_k.dtype)
store_v = dequantize_V(flat.value_q, group_size=engine.store.value_group_size).permute(1,0,2)
return torch.cat([store_k, ring_k], 0), torch.cat([store_v, ring_v], 0)
```
Flipping the toggle changed the output → proof the store is now in the causal path.

**Lesson**
Separate "the compressor ran" (storage) from "the model computed on compressed data"
(compute path). A storage metric can look great while the compute path is untouched. Prove
causality with a one-variable A/B switch.

---

## Case 8 — compression report shows `1.00x`

**Symptom**
First compression report: overall `1.00x`, "0 compressed".

**Why it happened**
The prompt produced a short answer (~15 tokens), well under `RING_CAPACITY=128`. The ring
never overflowed, so the store stayed empty — nothing was compressed. The 1.00x was correct
for that run, but it measured a regime the feature doesn't even engage in.

**How it was found**
The report prints the token split (`ring exact + store compressed`); seeing `0 compressed`
made it obvious the overflow path wasn't exercised.

**The fix** (force the regime you want to measure)
```python
MIN_NEW_TOKENS = 300   # push past RING_CAPACITY so the store fills
# generate(..., min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)
```
Then: seq 520, 8 exact + 512 compressed, overall 4.81x, compressed-segment 5.12x.

**Lesson**
A metric reading "no effect" often means the code path wasn't triggered, not that the effect
is zero. Make the test harness enter the regime under study before trusting the number.

---

## Case 9 — TQ-in-loop output degenerates (bug or bit budget?)

**Symptom**
With `USE_COMPRESSED_STORE=True`, output was coherent for ~130 tokens then collapsed into
`to to to...` / `AI AI AI...`.

**Why it happened (root cause)**
Two layers:
1. **Bit budget**: keys at 3 bits have large reconstruction error.
2. **Wrong compute path (the real one)**: `get_full_kv` does `dequantize -> dense SDPA`.
   TurboQuant is *designed* to score `⟨q, k⟩` directly from the quantized keys via an
   unbiased QJL estimator (`TurboQuant.attention_score(query, quantized_key)`). The
   estimator keeps *inner products* accurate even when point-wise key reconstruction is
   poor. Dequantizing first throws that away and feeds raw quant noise into attention, so
   the 0.43 key error hits the scores directly and compounds over 36 layers.

**How it was found** (the decisive sequence)
1. Round-trip identity test, isolated from the model:
```python
store.append_chunk(k, v); flat = store.get_flat_cache()
kr = store.quantizer.dequantize(flat.prod_q)        # decode(encode(k))
rel = (k - kr).norm() / k.norm()                    # relative error
```
   K3 key rel-err = 0.43, value 0.39 — high.
2. Precision sweep: K3V2 -> K4V4 -> K8V8 gave key 0.43 -> 0.23 -> 0.02. **Monotone, and
   K8 ≈ exact → the quantizer is correct; it's a budget/usage issue, not a code bug.**
3. Tried the obvious fix (K4/V4) and it *still* degenerated (3.02x). A failed fix is
   information: bits alone aren't the story.
4. `grep` for the store's methods surfaced `attention_score` / `attention_scores` — the
   intended fused path my wiring bypassed. That is the root cause.

**The fix (open)**
Route the compressed segment through the estimator instead of dequantize→dense:
```
ring scores  = q @ ring_k.T                                  # exact recent tokens
store scores = store.quantizer.attention_score(q, flat.prod_q)  # unbiased, quantized keys
logits = concat([store scores, ring scores]); softmax once
attend over [dequantize_V(store values) ⊕ ring values]
```
This is a rewrite of the patched attention (not `get_full_kv`). Raising bits is a stopgap;
the estimator path is the correct TurboQuant integration.

**Lesson**
Use the identity test + precision sweep to split bug from budget fast. When a plausible fix
doesn't help, the mental model is wrong — re-read the algorithm's intended interface; a
shortcut implementation often bypasses the exact mechanism that makes the method work. And
read the *shape* of the failure: "coherent then collapses, right when compressed history
takes over" pointed straight at the compressed-key compute path.

---

## Timeline at a glance

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `No module named 'src'` | script dir on sys.path, not repo root | insert project root on `sys.path` |
| 2 | `__init__ missing layer_classes` | 4.55 layered Cache init | guard `super().__init__()` |
| 3 | no attr `layers` (is_compileable) | base property reads `self.layers` | `is_compileable = False` |
| 4 | no attr `layers` (get_mask_sizes) | base delegates to `self.layers` | override `get_mask_sizes` |
| 5 | garbage `atsats...` | patch signature drift; RoPE from None | match 4.55 sig; use `position_embeddings` |
| 6 | decode still wrong | `is_causal=True` at q_len=1 | `is_causal = q_len==kv_len>1` |
| 7 | "is it really TQ?" | `get_full_kv` read ring only | wire store decompress + A/B toggle |
| 8 | report `1.00x` | ring never overflowed | `MIN_NEW_TOKENS` to force overflow |
| 9 | TQ-in-loop degenerates | dequantize→dense bypasses estimator (+ low bits) | route via `attention_score` (open) |
```
