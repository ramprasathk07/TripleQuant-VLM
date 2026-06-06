# 2026-06-06 — TurboQuant runtime config, model generalization, W&B logging

Branch: main · Status: uncommitted (working tree)

Two threads today: (1) finish + harden the TurboQuant KV-cache HF integration and
diagnose why low-bit keys produced garbage; (2) productize the benchmark runtime —
config-driven TurboQuant on/off, generalized model loading, and proper Weights &
Biases logging. Plus a pass over the user's runtime-package restructure.

---

## 1. TurboQuant KV-cache: root-caused the quality bug

Earlier in the session the TQ-in-loop generation degenerated at K3/V2. Findings
(written up in `notes/turboquant_kv_case_studies.md`, `debugging_turboquant_kv.md`):

- **Not a wiring bug.** K8/V8 generates cleanly → integration is correct; the loss is
  the quantizer's. Round-trip + bit sweep confirmed (K3 key rel-err 0.43, K8 0.02).
- **The QJL residual stage HURTS point-wise reconstruction.** It is an unbiased
  *inner-product* estimator (high variance at m=d projections); reused for key
  reconstruction it raises error at every bit width (K3 cos 0.94 → 0.92). The
  `attention_score` estimator is algebraically identical to dequantize-then-matmul
  (measured 0.437 == 0.437), so it gives no gain — a dead end.
- **Fix:** MSE-only key reconstruction (`use_qjl=False`). K3/V2 then generates
  coherently at ~5.12x compressed-segment / ~3.2x overall.
- Added `src/turboquant_v1/tests/test_quant_quality.py`: rel-err/cosine round-trip,
  MSE-vs-QJL, attention-score, value round-trip, store end-to-end, K8 sanity assert.
- Real algorithmic fix for low-bit keys (future): per-channel key quant (KIVI-style).
  Keys have outlier channels; the current per-vector codebook wastes the budget.

## 2. Checked the runtime-package restructure (user's refactor)

`src/runtimes/` split into `hf/` (hf_runtime + cache.py + patcher.py) and `vllm/`
subpackages; factory/__init__ rewired. Verified all critical hf_runtime fixes
survived (quantization_config repr patch for to_dict+to_diff_dict, `_model_load_kwargs`
omitting `quantization_config=None`, generation_config sanitize, VLM loader fallback,
measure_max_context). All compile + import clean; 3 benchmark configs dry-run valid.

Found the TQ-cache wiring was incomplete/broken — fixed in §3.

## 3. Productized the benchmark runtime (the main work)

### 3a. TurboQuant enable/disable + runtime wiring
- `src/config/schemas.py`: new `TurboQuantRuntimeConfig` (`enabled`, `ring_capacity`,
  `key_bits`, `value_bits`, `use_qjl`, `use_compressed_store`); `BenchmarkModelEntry`
  gains `turboquant: Optional[...]` + `turboquant_enabled` property.
- `src/runtimes/hf/config.py`: CREATED (was missing) — `CacheConfig` dataclass.
- `src/runtimes/hf/cache.py`: fixed broken imports (`from turboquant_v1` →
  `from src.turboquant_v1`) and removed invalid `use_qjl=` kwarg to `CompressedKVStore`.
- `HFRuntime`: `_maybe_enable_turboquant()` builds CacheConfig + patches attention +
  arms a per-call TQ cache; threaded into `generate`, `measure_ttft_tpot`,
  `measure_throughput` (single-sequence only — batched throughput falls back to the
  default cache since the TQ cache assumes batch=1). Disabled on VLMs; reset on unload.

### 3b. Generalized model loading
- `model_class` field on `BenchmarkModelEntry`:
  `auto | causal_lm | image_text_to_text | seq2seq_lm | vision2seq`. `auto` keeps the
  model_type heuristic; explicit forces the loader. `_resolve_load_route` maps it;
  `_load_causal_lm` now also handles seq2seq (AutoModelForSeq2SeqLM, right padding).
- `src/runtimes/hf/patcher.py`: rewritten to **duck-type** attention modules
  (`*Attention` + q/k/v/o + `layer_idx`) instead of hardcoding Llama/Qwen2/Mistral —
  now covers Qwen3/Gemma/Phi/etc. Added **QK-norm** (`q_norm`/`k_norm`) so Qwen3/Gemma2
  attention is numerically correct under TQ. Returns patched-count; warns+disables if 0.

### 3c. Production W&B logging
- `benchmark.py`: `_init_wandb` (one project, gated on `WANDB_API_KEY`/login so a
  no-creds run never blocks), `_flatten_metrics` (nested metrics → namespaced scalars,
  e.g. `perf/throughput/bs8/tokens_per_sec`, `quality_llm/gsm8k/acc`), and
  `_log_record_to_wandb` — **each `model @ runtime` is its own run** (`+tq` suffix when
  TurboQuant on), logs run config (model/class/dtype/quant/TQ/arch/status) + all metrics
  + run summary. Best-effort: tracking failure never aborts the benchmark.
- Installed `wandb` 0.27.2 + `python-dotenv` into `.venv_312`. `wandb login` to activate.
- `config/benchmark/qwen3_1-7b.yaml`: rewritten to demo `model_class`, a TurboQuant
  on/off pair, and a `tracking` block.

## Validation
- All touched files `py_compile` clean.
- Route resolver + `_flatten_metrics` unit-tested by hand.
- Schema accepts new fields (`model_class`, `turboquant`); CacheConfig defaults OK.
- All 4 benchmark configs `--dry-run` → "config valid ✓".
- NOT yet run end-to-end on real hardware (offered).

## Files touched
- `src/config/schemas.py` (model_class literal, TurboQuantRuntimeConfig, entry fields)
- `src/runtimes/hf/config.py` (new), `src/runtimes/hf/cache.py` (import fixes),
  `src/runtimes/hf/patcher.py` (generalized + QK-norm), `src/runtimes/hf/hf_runtime.py`
  (route resolution, TQ engagement, seq2seq, generate/measure threading)
- `benchmark.py` (wandb init + flatten + per-run logging + cleanup)
- `config/benchmark/qwen3_1-7b.yaml` (demo config)
- `src/turboquant_v1/tests/test_quant_quality.py` (new diagnostics)
- notes: `turboquant_hf_cache_guide.md`, `debugging_turboquant_kv.md`,
  `turboquant_kv_case_studies.md` (TQ understanding + debugging study guides)

## Open / next
- Run qwen3-1.7b config end-to-end (baseline vs +tq) on GPU to confirm.
- TurboQuant batched throughput (currently bs=1 only) — needs a batch-aware cache.
- Per-channel key quantization (KIVI) for usable low-bit keys.
- vLLM runtime: still single-request perf; async load harness is designed only
  (`notes/serving_sla_metrics.md`).
- MLflow tracker still unwired (TrackingConfig default lists it; only wandb implemented).
