# TripleQuant-VLM — Execution Plan (3-Week Sprint)

**Updated:** 2026-05-27
**Deadline:** 2026-06-12 (3 weeks)
**Time split:** Week 1 → quantization pipeline polish · Week 2 → TurboQuant (PyTorch reference) · Week 3 → kernel optimization (Triton)

Companion doc: `notes/turboquant.md` (algorithm + kernel deep-dive), `notes/benchmark.md` (eval/runtime/tracking).

---

## 1. Current State Snapshot (what changed since old plan)

### Done

| Component | File | Status |
|---|---|---|
| Pydantic v2 schemas | `src/config/schemas.py` | ✅ `QuantizeConfig` + family; `BenchmarkConfig`, `BenchmarkModelEntry` (+ runtime knobs `dtype`/`device_map`/`trust_remote_code`/`tensor_parallel_size`/`hf_quantization` + `is_vlm` property), `runtimes` selector, `MetricsConfig`, `EvalDatasetConfig`, `LatencyConfig`, `TrackingConfig`. |
| YAML loader | `src/config/loader.py` | ✅ `load_quantize_config` + `load_benchmark_config`. `src/config/__init__.py` re-exports both. |
| Registry | `src/quantization/registry.py` | ✅ Done. |
| BaseQuantizer | `src/quantization/base.py` | ✅ VLM/LLM dispatch via `model_type`. `_is_vlm`/`_get_vision_ignore_patterns`/`_merged_ignore` lifted to base. Save() builds descriptive subfolder. |
| LLM-Compressor adapter | `src/quantization/llm_compressor.py` | ✅ AWQ + GPTQ + PTQ + SmoothQuant via `_build_recipe`. Vision-tower ignore auto-merged. |
| ModelOpt adapter | `src/quantization/modelopt.py` | ✅ Renamed from `modelOpy.py`, **wired into factory**, rewritten for **modelopt 0.44 list-based `quant_cfg` API** (scheme→default-cfg map + AWQ variants + `block_sizes` patch + glob ignore). TinyLlama-1B AWQ-W4A16 runs (CPU-ext fallback on Windows). |
| Factory | `src/quantization/factory.py` | ✅ Routes on `backend`; ModelOpt branch active w/ lazy import + install hint. |
| Runtimes | `src/runtimes/{base,factory,hf_runtime,vllm_runtime}.py` | ✅ `RuntimeBase` contract; `HFRuntime` (logits/PPL/VLM/latency/throughput, left-padding) + `VLLMRuntime` (throughput/latency, logits unsupported). `build_runtime(name, entry)` reads `BenchmarkModelEntry` directly. |
| Eval — LLM | `src/evaluation/eval_llm.py` | ✅ `compute_ppl` (HF-only, logit-gated), `eval_mmlu_tiny`, `eval_logit_kl`, `eval_token_agreement`, `run_llm_eval`. |
| Eval — OCR | `src/evaluation/eval_ocr.py` | ✅ CER/WER/EM/BLEU, per-sample report, JSON export. Honors `dataset_name`. |
| Eval utils | `src/utils/{utils,hardware}.py` | ✅ HF dataset loaders, LaTeX normalize, prompt builder, GPU arch detect. |
| Benchmark entry | `benchmark.py` | ✅ Rewritten: dual-runtime loop, capability-routed metric groups, crash-safe per-(model,runtime) JSON + comparison summary, `skip_on` arch filter, `--dry-run`. |
| Entry point | `quantize.py` | ✅ load YAML → validate → factory → `load_model` → `quantize`. |
| Configs | `config/quantize/*.yaml`, `config/benchmark/{ocr,llm}_comparison.yaml` | ✅ quant configs + 2 benchmark configs (LLM + VLM/OCR). |

### Empty / missing

| Item | Status |
|---|---|
| `src/tracking/` | empty — `TrackingConfig` defined, no W&B/Langfuse/MLflow impl yet. Benchmark writes local JSON only. |
| `src/data/calibration.py`, `dataloader.py` | empty (calibration logic inline in adapters). |
| `src/quantization/fp16.py` | present, registered? (baseline identity quantizer — verify wired). |
| `arch_profiles.py` | not built — single-file dispatch in `llm_compressor.py`. |
| `eval_logit_kl` / `eval_token_agreement` in benchmark | stubbed in `benchmark.py` (`{"skipped": "requires baseline wiring"}`). Functions exist in `eval_llm.py`; need baseline-runtime capture loop. |
| Pruning / distillation / sparsity | not started (v2 scope). |
| TurboQuant | impl present in `src/turboquant/` but **`TMSE` broken** (see bugs). |

### Bugs / risks

1. ~~`loader.py` imports missing `BenchmarkConfig`~~ → **FIXED** (schema exists, `__init__` exports).
2. ~~`awq.py` dead code~~ → AWQ lives in `LLMCompressorQuantizer` + modelopt path.
3. ~~`modelOpy` NVFP4/MXFP4 hand-rolled dicts~~ → **FIXED**: modelopt 0.44 list-API, `mtq.*_CFG` defaults + patch.
4. ~~`modelOpy` factory branch commented~~ → **FIXED**: active.
5. ~~`OutputConfig` default~~ → **FIXED**: `output_dir` has default.
6. ~~`split` slice double-append~~ → **FIXED**: guarded `"[:" in split`.
7. **TurboQuant `TMSE` broken** (`src/turboquant/quantize.py`): buffer registered as `"RotateMatirx"` but `quantize`/`dequantize` reference undefined `self.Pi` → AttributeError. Owner: user (left alone per request).
8. **Windows: `modelopt_cuda_ext` won't JIT** (no MSVC `cl.exe`) → AWQ runs on slow CPU fallback. Install VS C++ Build Tools to enable.
9. **vLLM runtime untested on Windows** — vLLM dropped from `setup.bat` (torch-pin conflict). Run vLLM path in separate env.

### Remaining work (snapshot 2026-05-27)

Ordered by unblock value:

1. **Smoke-run the benchmark** end-to-end on TinyLlama (HF runtime) via `config/benchmark/llm_comparison.yaml` — confirm PPL + MMLU + ttft/tpot + throughput + memory all produce numbers and JSON saves. (Currently only `--dry-run` validated.)
2. **Wire `eval_logit_kl` + `eval_token_agreement`** into `benchmark.py`: when `config.baseline` set, load baseline runtime once, capture baseline logits/outputs on a fixed prompt set, pass into the metric fns. Currently stubbed as skipped.
3. **Trackers** (`src/tracking/`): implement W&B + MLflow + Langfuse loggers per `notes/benchmark.md §12b`; honor `TrackingConfig.enabled`; additive to local JSON. NoOp fallback when creds missing.
4. **fp16 baseline quantizer** — verify `fp16.py` is `@register`-ed and has a benchmark config (reference point for quality deltas).
5. **`arch_profiles.py`** (Week1 Day5) — `sequential_targets` + Mamba/MoE ignore patterns, wire into `llm_compressor._build_recipe`.
6. **TurboQuant `TMSE` fix** (user-owned) — `self.Pi` vs `RotateMatirx` buffer mismatch.
7. **vLLM path validation** in an isolated env (separate from modelopt env).

---

## 2. Three-Week Schedule

### Week 1 (May 22 → May 28) — Pipeline polish + benchmark + ModelOpt FP8

Goal: every config in `config/quantize/` runs end-to-end on both backends, produces a vLLM-loadable checkpoint, and a single eval script reports PPL + latency + file size.

**Status (May 27):** Day 1 blockers ✅ · Day 3 ModelOpt (0.44 API) ✅ · Day 4 benchmark harness ✅ (richer than planned — dual-runtime + OCR + perf + memory). Remaining: smoke-run benchmark, trackers, arch_profiles, fp16 baseline verify. See "Remaining work" above.

| Day | Task |
|---|---|
| Day 1 (Fri) | **Fix blockers.** `BenchmarkConfig` schema, `OutputConfig` default, delete dead `awq.py`, uncomment ModelOpt factory branch, `__init__.py` re-exports. |
| Day 2 | **fp16.py baseline.** Identity quantizer (`@register("fp16")`) — loads, saves, no modifier. Needed as benchmark reference. Add a YAML config for it. |
| Day 3 | **ModelOpt FP8 + NVFP4 wiring.** Replace hand-rolled NVFP4/MXFP4 dicts with `mtq.NVFP4_DEFAULT_CFG`, `mtq.MXFP4_DEFAULT_CFG`. Verify TinyLlama-1B FP8 end-to-end. If Blackwell available → NVFP4 too. |
| Day 4 | **Benchmark harness.** `benchmark.py` + `BenchmarkConfig` + `src/evaluation/` with three metrics: file size, perplexity (wikitext-2), single-prompt latency (tok/s). Crash-safe per-model loop. |
| Day 5 | **MoE sequential targets + Mamba ignore.** Add tiny `arch_profiles.py` that returns: `(sequential_targets, ignore_patterns)` keyed on `architectures[0]`. Wire into `llm_compressor.py` `_build_recipe` + `oneshot(sequential_targets=…)`. Skip pruning/distillation entirely. |
| Day 6 | **VLM eval add-on.** Extend `simple_generate.py` (or new `tests/eval_vlm.py`) to do image→text on a Qwen2.5-VL config and compute CER on 100 LaTeX_OCR samples. |
| Day 7 (Thu) | **Buffer day.** Re-run all 7 YAMLs, fix any breakage, commit clean checkpoint. |

**Out of scope this week:** AutoQuantize, SVDQuant, pruning, distillation, sparsity, TRT-LLM export. Document as "not implemented in v1" and move on.

### Week 2 (May 29 → June 4) — TurboQuant reference implementation (PyTorch)

Goal: working TurboQuant KV-cache quantizer end-to-end in pure PyTorch. Correctness > speed. See `notes/turboquant.md` for full math + algorithm.

| Day | Task |
|---|---|
| Day 8 | **Scaffold.** `src/turboquant/` package: `codebook.py` (Lloyd-Max), `rotation.py` (orthogonal + QJL matrices), `quantizer.py` (`TurboQuantMSE`, `TurboQuantProd`). Cache codebooks to disk JSON, keyed `(dim, bits)`. |
| Day 9 | **Bit-packing.** uint8 pack/unpack for 1/2/3/4-bit indices and QJL signs. Unit tests: round-trip random tensors at every bit width. |
| Day 10 | **Quantize/dequantize correctness.** Test on synthetic Gaussian vectors at `dim=128`: measure MSE vs uniform scalar quant; confirm TurboQuant_MSE beats per-coord min/max by ≥1.5×. Inner-product test: `TurboQuant_Prod` unbiased estimator within 2σ of true `<q,k>`. |
| Day 11 | **KV-cache capture.** Hook `LlamaAttention.forward` (or generic via `register_forward_hook`) to intercept K/V after RoPE, quantize, store in `KVStore`. Replace original cache with quantized handle. |
| Day 12 | **Dequant attention path (slow ref).** Override decode-step attention to: dequant K/V from store → standard `softmax(Q K^T / √d) V`. Validate generation quality on Llama-3-8B-Instruct: greedy decode N tokens, compare to FP16 baseline (token agreement %). |
| Day 13 | **Eval gate.** Run wikitext PPL on Llama-3-8B-Instruct with TurboQuant K=3-bit V=2-bit KV. Target: PPL delta < 0.3 vs FP16. Compare against vLLM FP8 KV baseline. |
| Day 14 (Thu) | **CLI + config.** YAML schema `TurboQuantConfig`, `@register("turboquant")` quantizer wrapper that ties into existing factory (note: TurboQuant is *runtime* compression, not weight quant — different code path; document the asymmetry). |

**Acceptance:** PyTorch reference passes correctness + accuracy gates. Slow (5–10× slower than FP16). That's expected — kernels next week.

### Week 3 (June 5 → June 12) — Triton kernel optimization

Goal: fused decode kernel that beats FP16 baseline on tok/s at long context (≥8K). Detailed kernel plan in `notes/turboquant.md` §5.

| Day | Task |
|---|---|
| Day 15 (Fri) | **Triton env check.** Confirm Triton version matches PyTorch, baseline `tl.dot` works on target GPU. Write trivial vector-add kernel as smoke test. |
| Day 16 | **Kernel 1 — MSE score (fused).** `<q_rot, centroids[idx]>` per K row. Inputs: packed indices `(N_k, packed_d)`, codebook `(2^b,)`, rotated query `(d,)`. Avoid materializing `k_hat`. Validate vs PyTorch. |
| Day 17 | **Kernel 2 — QJL score (fused).** `<S q, signs>` per K row. Pre-sketch query once outside the kernel. Unpack 8 signs per byte inline. Validate. |
| Day 18 | **Kernel 3 — fused decode (flash-style).** Combine: MSE score + QJL score + online softmax + value gather/dequant + weighted sum. One pass over KV. Tile on `N_k` axis. Validate. |
| Day 19 | **HF/vLLM shim.** Replace `LlamaAttention` decode-step with custom op when TurboQuant is active. For prefill, use FP16 + quantize *after*. Generate test sentence end-to-end. |
| Day 20 | **Benchmark.** Llama-3-8B, ctx 1K / 8K / 32K, batch 1: tok/s + peak VRAM, FP16 vs vLLM-FP8-KV vs TurboQuant 3-bit / 2-bit. Single chart. |
| Day 21 (Thu) | **Buffer + writeup.** Polish numbers, fix regressions, README section, demo recording. |

**Stretch goals (if Week 1 finishes early):** AutoQuantize wrapper, SVDQuant. Both are pure modelopt calls — low risk. Skip if blocking.

---

## 3. Final Repo Shape (after 3 weeks)

```
src/
  config/
    schemas.py        # ✅ Quantize* + Benchmark* + Metrics/Eval/Latency/Tracking
    loader.py         # ✅ load_quantize_config + load_benchmark_config
    __init__.py       # ✅ re-exports loaders + configs
  quantization/
    base.py           # ✅ shared VLM helpers lifted here
    registry.py       # ✅
    factory.py        # ✅ modelopt branch active
    fp16.py           # baseline identity (verify @register wired)
    llm_compressor.py # ✅ AWQ/GPTQ/PTQ/SmoothQuant
    modelopt.py       # ✅ FP8/NVFP4/MXFP4/INT — modelopt 0.44 list-API
    arch_profiles.py  # TODO (week1 day5) — sequential_targets + ignore
  runtimes/           # ✅ NEW — dual-runtime benchmarking
    base.py           # RuntimeBase contract
    factory.py        # build_runtime(name, entry)
    hf_runtime.py     # HF: logits/PPL/VLM/latency/throughput
    vllm_runtime.py   # vLLM: throughput/latency (no logits)
  evaluation/         # ✅ NEW
    eval_llm.py       # PPL, MMLU, logit-KL, token-agree
    eval_ocr.py       # CER/WER/EM/BLEU + per-sample report
  utils/              # ✅ NEW
    utils.py          # dataset loaders, LaTeX norm, prompt builder
    hardware.py       # GPU vendor/arch/VRAM detect
  tracking/           # ⬜ empty — W&B/Langfuse/MLflow TODO
  turboquant/         # ⚠️ present; TMSE broken (self.Pi) — user-owned
    quantize.py kv_cache.py lloyd_codebook.py rotations.py memory.py
  data/               # ⬜ calibration.py/dataloader.py empty (inline for now)
benchmark.py          # ✅ rewritten — dual-runtime, crash-safe, summary
quantize.py           # ✅
config/
  quantize/*.yaml
  benchmark/ocr_comparison.yaml   # ✅ VLM/OCR
  benchmark/llm_comparison.yaml   # ✅ text LLM (ppl+mmlu+perf+mem)
notes/
  plan.md  benchmark.md  kernel_scope.md  turboquant.md
```

---

## 4. Compatibility Guards (still load-bearing)

| Failure | Guard |
|---|---|
| AWQ + non-INT4 scheme | Pydantic cross-validator (already in `schemas.py`) |
| VLM vision tower quantized | `_get_vision_ignore_patterns` auto-merge in both adapters |
| Mamba SSM quantized | New: add `re:.*conv1d.*`, `re:.*A_log.*`, `re:.*dt_proj.*`, `re:.*x_proj.*` to `arch_profiles.py` |
| MoE OOM | Pass `sequential_targets` to `oneshot` per arch profile |
| NVFP4 on non-Blackwell | Hardware floor check at start of `ModelOptQuantizer.quantize` |
| ModelOpt missing | Already lazy import — wrap in try/except with `pip install nvidia-modelopt[torch]` hint |
| TurboQuant prefill | Document: TurboQuant compresses *after* prefill; quantize K/V at the end of prefill, then decode reads compressed. |
| TurboQuant + GQA | Codebook is per-`head_dim` (typically 128 for Llama). KV-heads count is orthogonal. Confirm in unit test. |

---

## 5. What we are explicitly *not* doing (deferred past June 12)

- Minitron pruning (depth + width)
- Knowledge distillation post-prune (paper recipe is 80–100B tokens — no compute budget)
- AutoQuantize per-layer search
- SVDQuant
- 2:4 sparsity
- TRT-LLM export
- Audio modality calibration
- KV-cache eviction policies (H2O, StreamingLLM)
- NeMo/Megatron containers

Document each in README as "v2 scope" with one-line rationale. Don't half-implement.

---

## 6. Open questions (please answer before Day 1)

1. **Target GPU(s)?** Ampere (A100/3090) → INT4/INT8 + TurboQuant in fp32 fallback. Hopper (H100) → add FP8 path. Blackwell (B100/RTX 50) → NVFP4 in scope. Tells us whether NVFP4 stays in Week 1.
2. **VLM accuracy target** — CER ≤ baseline + 2pp on LaTeX_OCR? Or just "doesn't crash"? Drives how much calibration tuning matters in Week 1.
3. **TurboQuant deployment target** — HF `generate` only (simpler), or also vLLM v0 attention backend (harder, may push Week 3)? Default assumption: HF only; vLLM stretch.
4. **Acceptable PPL delta** for TurboQuant Week 2 gate? Paper claims neutrality at 3.5 bits, marginal at 2.5. Suggest gate: ≤ +0.3 wikitext PPL at K=3/V=2.
5. **W&B / tracking** — do you want runs logged, or local JSON only? `src/integrations/wandb_logger.py` is empty.
