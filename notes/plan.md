# TripleQuant-VLM — Execution Plan (3-Week Sprint)

**Updated:** 2026-05-22
**Deadline:** 2026-06-12 (3 weeks)
**Time split:** Week 1 → quantization pipeline polish · Week 2 → TurboQuant (PyTorch reference) · Week 3 → kernel optimization (Triton)

Companion doc: `notes/turboquant.md` (algorithm + kernel deep-dive).

---

## 1. Current State Snapshot (what changed since old plan)

### Done

| Component | File | Status |
|---|---|---|
| Pydantic v2 schemas | `src/config/schemas.py` | ✅ Done — `QuantizeConfig`, `SchemeConfig`, `CalibrationConfig`, `OutputConfig`, `SmoothQuantConfig`, `AWQParams`, `GPTQParams`. Method-vs-scheme cross-validator works. |
| YAML loader | `src/config/loader.py` | ⚠️ Imports `BenchmarkConfig` but symbol does not exist in `schemas.py`. **Import error at runtime.** |
| Registry | `src/quantization/registry.py` | ✅ Done. |
| BaseQuantizer | `src/quantization/base.py` | ✅ VLM/LLM dispatch via `model_type` (no string sniff). Save() builds descriptive subfolder. |
| LLM-Compressor adapter | `src/quantization/llm_compressor.py` | ✅ AWQ + GPTQ + PTQ + SmoothQuant unified through `_build_recipe`. Vision-tower ignore auto-merged. Dataset auto-detect (chat vs image-text). |
| ModelOpt adapter | `src/quantization/modelOpy.py` | ⚠️ Exists, **not wired into factory** (commented out). NVFP4/MXFP4 use hand-rolled config dicts, not `mtq.NVFP4_DEFAULT_CFG`. Untested. |
| Factory | `src/quantization/factory.py` | ✅ Routes on `backend`. ModelOpt branch commented out. |
| AWQ class | `src/quantization/awq.py` | ⚠️ Stub only — no `quantize()` body. Actual AWQ now lives inside `LLMCompressorQuantizer`. **Class is dead code; delete or fold.** |
| Entry point | `quantize.py` | ✅ Wired: load YAML → validate → factory → `load_model` → `quantize`. |
| Configs | `config/quantize/*.yaml` | ✅ 7 configs: TinyLlama-1B (llmcomp+modelopt), Qwen2.5-VL-3B/7B (awq/gptq), SmolVLM2, nanoLLaVA. |
| Generate/eval test | `tests/simple_generate.py` | ✅ Loads compressed model, prompts, prints VRAM/throughput. LLM only (no VLM eval, no PPL, no CER/WER). |

### Empty / missing

| Item | Status |
|---|---|
| `src/quantization/__init__.py`, `fp16.py` | empty |
| `src/config/__init__.py`, `src/__init__.py`, `src/data/__init__.py`, `src/integrations/__init__.py` | empty |
| `src/data/calibration.py`, `dataloader.py` | empty (logic lives inline in `llm_compressor.py`) |
| `src/integrations/wandb_logger.py` | empty |
| `BenchmarkConfig` schema | not defined (breaks `loader.py`) |
| `benchmark.py` entry | not present |
| `arch_profiles.py`, `schemes.py` | not built — single-file dispatch in `llm_compressor.py` |
| Pruning / distillation / sparsity | not started |
| TurboQuant | only enum value `"turboquant"` in `MethodLiteral` — no impl |

### Bugs / risks to fix before adding new code

1. `loader.py` imports `BenchmarkConfig` which doesn't exist → any `from src.config.loader import …` will explode. Either delete `load_benchmark_config` or add `BenchmarkConfig` (matches `BenchmarkModelEntry` already in schemas).
2. `awq.py` is dead. Remove or implement `quantize()` to call `LLMCompressorQuantizer` — current state is a trap.
3. `modelOpy.py` `_get_nvfp4_cfg` / `_get_mxfp4_cfg` build dicts that `mtq.quantize` will reject. Use `mtq.NVFP4_DEFAULT_CFG.copy()` and override fields, like the INT4/INT8 branches do.
4. `modelOpy.py` factory branch commented — `backend: modelopt` configs currently raise `ValueError`.
5. `OutputConfig(output_dir: Path)` has no default, but `QuantizeConfig.output: OutputConfig = Field(default_factory=OutputConfig)` will fail validation when YAML omits `output`. Either make `output_dir` `Optional` or drop the default factory.
6. `_load_calibration_dataset` in `llm_compressor.py` passes `split=f"{split}[:{n}]"` — fine for HF but breaks when YAML user already added a slice.

---

## 2. Three-Week Schedule

### Week 1 (May 22 → May 28) — Pipeline polish + benchmark + ModelOpt FP8

Goal: every config in `config/quantize/` runs end-to-end on both backends, produces a vLLM-loadable checkpoint, and a single eval script reports PPL + latency + file size.

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
    schemas.py        # + BenchmarkConfig, + TurboQuantConfig
    loader.py
  quantization/
    base.py
    registry.py
    factory.py
    fp16.py           # baseline (NEW)
    llm_compressor.py # AWQ/GPTQ/PTQ/SmoothQuant
    modelOpy.py       # FP8/NVFP4/MXFP4 (FIXED)
    arch_profiles.py  # tiny — sequential_targets + ignore (NEW)
  turboquant/         # NEW — week 2-3
    __init__.py
    codebook.py       # Lloyd-Max + cache
    rotation.py       # orthogonal Π + QJL S
    quantizer.py      # TurboQuantMSE / TurboQuantProd
    pack.py           # bit-packing helpers
    kv_cache.py       # capture hooks + compressed store
    attention.py      # Python ref decode-step attention
    triton_kernels.py # NEW week 3
    integration/
      hf_llama.py     # monkey-patch shim
  evaluation/         # NEW — week 1
    metrics.py        # PPL, latency, VRAM, file size
    ocr_cer.py        # VLM accuracy (Qwen2.5-VL)
  data/
    calibration.py    # extracted from llm_compressor.py
benchmark.py          # NEW
quantize.py           # existing
notes/
  plan.md             # this file
  turboquant.md       # algorithm + kernel deep-dive
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
