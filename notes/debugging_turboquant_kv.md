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

### 3.2 Verify a proposed fix's premise before building it
Raising to K4/V4 *still* degenerated, so I formed a hypothesis: "the wiring uses
`dequantize → dense SDPA` but TurboQuant exposes `attention_score(query, quantized_key)` —
an estimator that should keep inner products accurate at low bits. Use that instead."
- **Technique**: before implementing a fix, *measure its premise*. I compared the estimator
  scores against dequantize→matmul scores on the same data:
  ```python
  s_est = store.quantizer.attention_score(q, flat.prod_q)
  s_dq  = q @ store.quantizer.dequantize(flat.prod_q).transpose(-2, -1)
  # rel-err vs true q@k.T: estimator 0.437, dequant 0.437 -> IDENTICAL
  ```
- **What it found**: they are the same number. The estimator is *algebraically equal* to
  dequantize-then-matmul for a fixed QJL projection S (its benefit is variance reduction in
  expectation over random S, not a per-call accuracy gain). The hypothesized fix was a dead
  end — and I learned that in one cheap measurement instead of a multi-hour rewrite.
- **Takeaway**: an appealing fix can be wrong. When a fix rests on "method X is more
  accurate," test that claim numerically *first*. Building it then measuring wastes the most
  time.

### 3.3 The decisive test: clean high-bit run rules out a bug
- **Technique**: to settle "bug vs budget" definitively, push the lossy knob to near-lossless
  and see if the symptom disappears. Ran generation at K8/V8.
- **Result**: K8/V8 produced fully coherent text (proper ending). A clean run at high bits
  proves there is *no integration bug* — every wiring step is correct; the only variable that
  matters is fidelity. K8/V4 then gave clean output at a usable ~2.25x segment ratio.
- **Why this beats more reasoning**: it's a single experiment that collapses the hypothesis
  space. If high-bit had *also* degenerated, the bug would be in the integration, not the
  codec. It didn't, so it's the codec/budget.
- **Takeaway**: when unsure whether a defect is "wrong code" or "aggressive setting," set the
  setting to its safe extreme. Symptom gone => setting; symptom stays => code.

### 3.4 Causal reasoning for the *shape* of the failure
The output was coherent for a while, then collapsed. That pattern is itself a clue.
- **Reasoning**: with an exact ring of recent tokens, recent context is fine; degradation
  begins precisely when attention must rely on the compressed tail, and **errors compound
  across 36 layers and hundreds of positions** (softmax over many noisy logits amplifies
  small per-key errors). The *location and progressiveness* matched the
  compressed-history-fidelity hypothesis.
- **The deeper why**: keys need ~8 bits here because the quantizer treats keys per-vector
  with no outlier handling; transformer keys have large outlier *channels*. The KIVI fix is
  per-channel key quantization. That's the real path to low-bit keys — a quantizer change,
  not a wiring change.
- **Takeaway**: don't just ask "does it fail" — ask "*where* and *how*, and is that
  consistent with my hypothesis?" The failure's shape is evidence.

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
- `notes/turboquant_kv_case_studies.md` — bug-by-bug study guide (this journal's companion).
- `notes/turboquant.md` — the TurboQuant algorithm.
- Applied fix: `KEY_BITS=8, VALUE_BITS=4` gives clean generation at ~2.25x segment. The real
  algorithmic fix for low-bit keys is per-channel key quantization (KIVI-style), a quantizer
  change — not the `attention_score` estimator, which is the same math as dequant (Phase 3.2).
