# Engineering Notes

This is an index, not a duplicate — the full detail lives in `notes/` (methodology
journal, architecture walkthrough, case-by-case bug studies). This page pulls the
transferable lessons up front and points to where the full story is.

---

## Building the TurboQuant HF integration

`src/turboquant_v1/tests/hf_cache.py` swaps a stock HuggingFace model's KV cache for a
custom TurboQuant cache and monkeypatches attention to read from it. It was written
against a pre-4.55 `transformers`; getting it working on 4.55's refactored `Cache` +
attention APIs took nine distinct, documented bugs across three phases:

1. **Make it run** — a chain of framework API mismatches (`Cache.__init__` signature
   change, missing `self.layers`, attention `forward` signature drift).
2. **Make it honest** — it produced text with no exceptions, but was TurboQuant actually
   in the compute path, or was it silently falling back to an exact sliding window?
3. **Make it correct** — it was genuinely using TurboQuant, but output degenerated at
   low bit-widths — bug, or expected loss at that budget?

Full case-by-case detail: [`notes/turboquant_kv_case_studies.md`](../notes/turboquant_kv_case_studies.md)
(9 bugs, each with symptom / root cause / how it was found / before-after fix / lesson).

Full methodology narrative (the *why* behind each technique, not just the fix):
[`notes/debugging_turboquant_kv.md`](../notes/debugging_turboquant_kv.md).

Architecture walkthrough (how the two-tier ring+store cache actually works, RoPE/GQA
interaction, the `USE_COMPRESSED_STORE` toggle, compression-ratio measurement):
[`notes/turboquant_hf_cache_guide.md`](../notes/turboquant_hf_cache_guide.md).

### The reusable toolkit (condensed from the methodology journal)

1. **Read the traceback bottom-up; fix the exact frame.** Open the *library* source at
   the failing line to learn why, not just patch the symptom.
2. **Environment vs. code.** Import/path errors are almost always `sys.path` or
   environment, not logic — the tell is "works with `-m`, fails as a direct script."
3. **Framework-subclass fix-rerun loop.** Exceptions from subclassing a large framework
   class enumerate the API contract the running version actually requires. Prefer correct
   overrides over plausible stubs; watch empty-collection/vacuous-truth edge cases when
   stubbing (`all()` over `[]` is `True`).
4. **Runs != correct.** Treat output quality as a test signal, not just exit code. Reason
   explicitly about masking/shapes that never raise (e.g. `is_causal=True` at decode
   `q_len=1` silently attends to nothing but key 0 — no exception, just wrong math).
5. **Pin monkeypatch signatures** against the live framework with `inspect.signature`.
   Silent arg-slot drift from a keyword-only call is a classic cause of silent-wrong.
6. **Be suspicious of your own success.** Find the line that proves the claim (a `# TODO`
   in `get_full_kv` revealed the "TurboQuant" run was actually reading an exact bf16
   ring). Build a one-variable A/B toggle to prove causality, not just plausibility.
7. **Distinguish storage vs. compute path.** A compressor can be genuinely shrinking
   storage while the model never reads the compressed values — measure the one your
   metric actually reflects. This exact class of bug recurred at the benchmark level
   months later (see below).
8. **Identity (round-trip) test + precision sweep** to separate bug from budget. Monotone
   error-vs-precision is the signature of a correct codec at a tight budget; a flat or
   non-monotone curve signals an actual bug.
9. **Verify a fix's premise numerically before building it.** The QJL-estimator "fix" for
   low-bit key quality looked plausible; one measurement showed it's algebraically
   identical to dequantize-then-matmul (see `failure_cases.md` #3) — a multi-hour rewrite
   avoided by a five-line check.
10. **Interpret the failure's shape.** *Where* and *how* something breaks is evidence —
    coherent-then-collapsing output pointed straight at compressed-tail fidelity, not a
    wiring bug, before any measurement confirmed it.

---

## A second instance of the same lesson: the benchmark harness itself

Lesson #7 above ("storage vs. compute path — measure the one your metric reflects")
recurred independently at a different layer of the system, a month later: perplexity
looked bit-identical between FP16 and TurboQuant not because TurboQuant was lossless, but
because teacher-forced PPL never reads the KV cache at all (`failure_cases.md` #4). Same
underlying mistake — trusting that a metric reflects a mechanism without checking whether
the metric's code path actually exercises that mechanism — caught the same way: reasoning
about what the metric computes, then designing a measurement (generation, not
teacher-forcing) that actually exercises the path in question.

A third instance: the Windows VRAM-oversubscription bug (`failure_cases.md` #1) was
"no exception raised" being trusted as "it fits," structurally the same shape as
lesson #4 above ("runs != correct" — CUDA's silent-success-via-paging is just another
form of no-exception-but-wrong).

---

## Kernel optimization scope (not yet built)

TurboQuant currently runs on a pure PyTorch reference path — correct, unfused, 5-10x
slower than a fused decode kernel would be. The kernel plan (Triton for the
TurboQuant-specific kernels; TileLang scoped as a post-v1.0 option for cross-vendor fused
QK-V) is fully designed but not implemented:

- Math + Triton pseudocode for all four candidate kernels (MSE score, QJL score, fused
  decode, fused prefill quantize): [`notes/turboquant.md`](../notes/turboquant.md) §5.
- Full kernel-by-kernel priority/difficulty/speedup survey, plus a Triton-vs-TileLang
  decision framework and TileLang learning path:
  [`notes/kernel_scope.md`](../notes/kernel_scope.md).
- **Where to start if you're new to kernels**:
  [`docs/kernel_learning_path.md`](kernel_learning_path.md) — the same material ordered
  easiest-to-hardest, with what each kernel actually does and where it plugs into this
  repo, anchored to the measured 13x HF-eager-vs-vLLM gap rather than generic advice.

Status tracked in `failure_cases.md` #8 and the README roadmap.
