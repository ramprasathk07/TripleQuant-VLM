# Debugging journal: bringing the TurboQuant KV-cache up on transformers 4.55

This is a methodology guide, not a changelog. It records *how* the TurboQuant
`hf_cache.py` integration was debugged — the techniques, the order they were applied,
and the reasoning that picked each one. The goal is that you can reuse the methods on the
next framework-integration bug, not just memorize these fixes.

The work went through three phases, each needing a different debugging mindset:
1. **Make it run** — a chain of `AttributeError`/`TypeError` from a framework API that
   changed under the code.
2. **Make it honest** — it ran and produced text, but was it actually using TurboQuant?
3. **Make it correct** — it used TQ, but the output degenerated; bug or expected?

Each phase below: the symptom, the technique, why that technique, and the takeaway.

---

## Phase 1 — make it run (read tracebacks, fix the exact frame)

### 1.1 `ModuleNotFoundError: No module named 'src'`
- **Technique**: read the *first* line of the traceback literally, and reason about the
  runtime context, not the code. The import is correct; what's wrong is `sys.path`.
- **Why**: `import` errors are almost never about the imported module's content — they're
  about search path. The tell: it works as `-m` but not as a direct script. Running
  `python a/b/c.py` puts `a/b/c/` on `sys.path[0]`, not the repo root.
- **Fix shape**: insert the project root (`Path(__file__).resolve().parents[N]`) on
  `sys.path` before the first-party imports.
- **Takeaway**: separate "is the code wrong" from "is the environment wrong". Import and
  path errors are usually the latter.

### 1.2 / 1.3 / 1.4 — the framework-subclass error walk
Symptoms, in the order they appeared, all from subclassing `transformers.Cache`:
- `Cache.__init__() missing 1 required positional argument: 'layer_classes'`
- `'TurboDynamicCache' object has no attribute 'layers'` (inside `is_compileable`)
- same, inside `get_mask_sizes`

- **Technique**: the **fix-rerun loop**. When you subclass a large framework class, the
  exceptions become a checklist: each rerun surfaces the next method the framework calls
  that your subclass doesn't satisfy. Fix the exact failing access, rerun, repeat.
- **Why this and not "read all the docs first"**: the framework (transformers 4.55) had
  just refactored `Cache` into a layered design. Docs lag refactors; the *running code* is
  ground truth about what's actually called on your object. Let the tracebacks enumerate
  the contract.
- **Reasoning for each fix**: read the failing frame in the *library* source
  (`cache_utils.py:1286` for `is_compileable`, `:1219` for `get_mask_sizes`). Each read
  showed the base implementation iterating `self.layers`, which a from-scratch cache
  doesn't have. So override that one access point with a direct computation, rather than
  trying to fake a `layers` list (faking it risks wrong behavior elsewhere — see "vacuous
  truth" note below).
- **Subtlety caught**: `is_compileable` is a `@property` reading `self.layers`. Setting
  `self.layers = []` would make `all(... for ... in [])` return `True` (vacuous truth) →
  transformers would try to `torch.compile` an incompatible cache. So override to a hard
  `False` instead. *Lesson: when stubbing, think about the empty-collection edge case.*
- **Takeaway**: framework integration errors are a guided tour of the API contract. Read
  the library frame, fix the precise access, prefer correct overrides over plausible stubs.

---

## Phase 2 — make it honest (runs != correct; verify the claim)

After Phase 1 the script generated coherent text. Easy to declare victory. Two traps were
avoided here.

### 2.1 Garbage output — "runs" is not "works"
At one point it generated `atsatsats...`. Mechanically fine (no exception), semantically
broken.
- **Technique**: treat *output quality* as a test signal, not just exit code. Then localize
  by comparing the patched function's signature against the framework's *current* expected
  signature with `inspect.signature(Qwen2Attention.forward)`.
- **Why**: the patch was monkeypatching `Qwen2Attention.forward`. If the host signature
  changed, the patch silently receives args in the wrong slots. Printing the live signature
  is the fastest way to see the mismatch instead of guessing.
- **What it revealed**: 4.55 passes everything by keyword, `position_embeddings` replaced
  `position_ids`, and there is no `use_cache` kwarg. The old patch gated on `use_cache`
  (never set) so the TQ branch never ran, and it recomputed RoPE from a `None`
  `position_ids`. Two bugs, both from signature drift.
- **Second-order bug found by reasoning, not by error**: `is_causal=True` in SDPA. At
  decode `q_len==1`, kv_len==T; a causal mask aligns the single query to key 0 only. No
  exception — just wrong math. Caught by thinking about *what the mask means* at q_len=1,
  not by a stack trace.
- **Takeaway**: silent-wrong is worse than loud-wrong. When monkeypatching, pin the
  signature against the live framework, and reason explicitly about masking/shapes that
  never raise.

### 2.2 "Are you sure it used TQ, not the baseline cache?"
The right question to be suspicious of your own success.
- **Technique**: **isolate the claim and find the line that proves or disproves it.** I read
  `get_full_kv` and found it returns `ring.peek()` only, with a literal
  `# TODO: prepend compressed tokens`. So the compressed store was write-only; attention
  read exact bf16. The "TQ" result was a bf16 sliding window.
- **Then: prove causality with an A/B toggle.** Added `USE_COMPRESSED_STORE`. False = ring
  only (clean output). True = decompress store into attention (output changes/degrades).
  If flipping a switch changes the result, you've *proven* the switch is in the causal path.
- **Why a toggle beats argument**: claims about "is X actually happening" are settled by a
  controlled experiment, not by reading code alone. The toggle is a one-variable A/B.
- **Takeaway**: distinguish "the quantizer ran" (storage) from "the model computed on
  quantized data" (compute path). Verify which one your metric reflects. Build the A/B
  switch to make causality testable.

### 2.3 Measuring the thing you claim (compression report)
- **Technique**: write the metric so it states its own assumptions. The report separates
  "overall" (ring bf16 + store packed) from "compressed-segment only", and prints the
  token split, because a single ratio hides *why* it is what it is.
- **Trap caught**: first run reported 1.00x — because a short answer never overflowed the
  128-token ring, so the store was empty. Added `MIN_NEW_TOKENS` to *force* the regime you
  intend to measure. *Lesson: a metric that reads "no effect" often means the code path
  wasn't exercised, not that the effect is zero. Make the harness enter the regime.*

---

## Phase 3 — make it correct (bug vs budget; question the abstraction)

TQ was now genuinely in the loop, but output degenerated into repetition once attention
leaned on compressed history. The key question: **is this a bug, or expected loss at these
bits?** Spending time fixing a "bug" that's actually a parameter choice is wasted; shipping
a real bug as "expected" is worse. So decide which, cheaply, before doing more work.

### 3.1 Round-trip test to separate bug from budget
- **Technique**: the **identity test**. For any lossy codec, `decode(encode(x))` should
  approach `x` as precision rises. Quantize then dequantize a controlled tensor, measure
  relative error and cosine similarity.
- **Why**: it removes the entire model from the loop. If the round-trip is bad even in
  isolation, the quantizer is suspect. If it's good, the problem is downstream (wiring,
  bits, or compounding).
- **Why a parameter sweep on top**: ran it at K3/V2, K4/V4, K8/V8. Error fell
  monotonically (key 0.43 → 0.23 → 0.02). **Monotone-with-precision is the signature of a
  correct codec at a tight budget; a flat/garbage curve would signal a bug.** K8 ≈ exact
  confirmed the implementation is sound.
- **Inputs chosen deliberately**: Gaussian vectors — a structureless worst case. Real K/V
  have structure quantizers exploit, so Gaussian gives a pessimistic floor. Knowing whether
  you're testing best/typical/worst case is part of the method.
- **Takeaway**: the identity test + precision sweep is the fastest bug-vs-budget
  discriminator for any compressor. Pick inputs whose difficulty you understand.

### 3.2 The fix that didn't work is information
Raising to K4/V4 *still* degenerated. A failed fix is a probe: it falsifies "low bits are
the whole story" and forces a model-level rethink.
- **Technique**: when a plausible fix doesn't move the needle, stop tuning and re-examine
  assumptions. Re-read the algorithm's intended interface.
- **What it found**: `grep` for the methods the store exposes surfaced
  `TurboQuant.attention_score(query, quantized_key)` and `attention_scores` on the cache —
  i.e. TurboQuant is *designed* to compute ⟨q, k⟩ directly from the quantized keys via an
  unbiased QJL estimator. My `get_full_kv` did `dequantize → dense SDPA`, which throws that
  estimator away and feeds raw reconstruction noise into attention.
- **Why this is the root**: the dequantize-then-matmul path makes the model see the codec's
  point-wise error; the estimator path is built to keep the *inner products* accurate at low
  bits even when point-wise reconstruction is poor. Using the wrong path means the 0.43
  key error hits attention directly — exactly the observed collapse.
- **Takeaway**: when fixes within a mental model fail, the mental model is probably wrong.
  Re-read the library's intended usage; a "shortcut" implementation often bypasses the very
  mechanism that makes the method work.

### 3.3 Causal reasoning for the *shape* of the failure
The output was coherent for ~130 tokens, then collapsed. That pattern is itself a clue.
- **Reasoning**: with a 128-token exact ring, the most recent context is fine; degradation
  begins precisely when attention must rely on the compressed tail, and **errors compound
  across 36 layers and hundreds of positions** (softmax over many noisy logits amplifies
  small per-key errors). The *location and progressiveness* of the failure matched the
  hypothesis (compressed-history dependence), which corroborated the root cause.
- **Takeaway**: don't just ask "does it fail" — ask "*where* and *how* does it fail, and is
  that consistent with my hypothesis?" The failure's shape is evidence.

---

## The reusable toolkit (distilled)

1. **Read the traceback bottom-up; fix the exact frame.** Open the *library* source at the
   failing line to learn *why*, not just patch the symptom.
2. **Environment vs code.** Import/path errors are usually `sys.path`/env, not logic.
3. **Framework-subclass fix-rerun loop.** Exceptions enumerate the API contract the running
   version actually requires; let them. Prefer correct overrides over plausible stubs, and
   check empty-collection / vacuous-truth edge cases when stubbing.
4. **Runs != correct.** Use output quality as a test signal. Reason explicitly about
   math that never raises (masks, shapes, dtypes).
5. **Pin monkeypatch signatures** against the live framework with `inspect.signature`.
   Silent arg-slot drift is a classic cause of silent-wrong.
6. **Be suspicious of your own success.** Find the line that proves the claim. Build an A/B
   **toggle** to test causality with one variable.
7. **Distinguish storage vs compute path.** Measure the one your metric actually reflects;
   state the metric's assumptions; force the code into the regime you mean to measure.
8. **Identity (round-trip) test + precision sweep** to separate bug from budget. Monotone
   error-vs-precision = sound codec at a tight budget. Choose inputs of known difficulty.
9. **A failed fix is data.** It falsifies a hypothesis; let it push you up an abstraction
   level. Re-read the intended interface — shortcuts often bypass the key mechanism.
10. **Interpret the failure's shape.** *Where* and *how* it breaks should match your root-
    cause hypothesis; if it doesn't, the hypothesis is incomplete.

---

## Cross-references
- `notes/turboquant_hf_cache_guide.md` — what the script does, end to end.
- `notes/turboquant.md` — the TurboQuant algorithm and the intended fused score path.
- The open fix: route the compressed segment through `quantizer.attention_score()` instead
  of `dequantize → dense SDPA` (see Phase 3.2).
