# Unified Quantization Layer — Implementation Plan

## 1. Goal

Refactor `src/quantization/` (and add `src/pruning/`, `src/distillation/`) into a model-agnostic, precision-agnostic compression layer that handles:

- **Model families:** dense LLM, VLM (vision-language), MoE (mixture-of-experts), Mamba / state-space, hybrid (Jamba, Zamba, Falcon-Mamba).
- **Quantization methods:** AWQ (INT4-only by definition), GPTQ, PTQ / RTN, SmoothQuant pre-pass, SVDQuant (modelopt), AutoQuantize (modelopt).
- **Precisions:** INT8 (W8A8 / W8A16), INT4 (W4A16, W4A8, W4A4), FP8 (E4M3, E5M2, per-tensor, per-channel/per-token, block `FP8_PB_WO`), NVFP4 (Blackwell, block-size 16) incl. NVFP4-MLP / NVFP4-experts variants, MXFP4 (OCP microscaling, block-size 32), MXFP8.
- **Pruning:** Minitron depth (`num_layers`) + width (`hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `num_kv_heads`, `mamba_num_heads`, `mamba_head_dim`, `num_moe_experts`, `moe_ffn_hidden_size`). FastNAS for CV.
- **Distillation:** mandatory post-pruning recovery (Minitron pipeline) — pair teacher (original) + student (pruned) over 80–100B tokens.
- **Sparsity (optional, later):** 2:4 structured sparsity (Ampere+) via modelopt `mts.sparsify`.

**Dual backend strategy:**

| Backend | Strengths | Use when |
|---|---|---|
| `llmcompressor` + `compressed-tensors` | Mature HF integration, broad scheme presets, AWQ + GPTQ + SmoothQuant unified, vLLM-native | Default for LLM/VLM INT4/INT8/FP8 |
| `nvidia-modelopt` (`modelopt.torch.quantization` / `.prune` / `.distill` / `.export`) | NVFP4 / MXFP4 / MXFP8 / SVDQuant / AutoQuantize / Minitron pruning / KV-cache quant; TRT-LLM + vLLM + SGLang export | Required for NVFP4/MXFP4/MXFP8, pruning, distillation, AutoQuantize per-layer search |

vLLM consumes both: `compressed-tensors` (llmcompressor) and `modelopt` / `modelopt_fp4` / `modelopt_mxfp8` (via `hf_quant_config.json`).

## 2. Current State (2026-05-20)

| File | Status |
|---|---|
| `src/quantization/base.py` | Present — has VLM/LLM loader, save helper |
| `src/quantization/awq.py` | Present — Qwen2.5-VL biased, layer-count auto-detect stub |
| `src/quantization/registry.py` | Present — decorator registry |
| `src/quantization/__init__.py` | **Empty** |
| `src/quantization/fp16.py` | **Empty** |
| `src/quantization/gptq.py` | **Deleted** (was in git, removed) |
| `src/config/schemas.py` | **Empty** — `QuantizeConfig` referenced but undefined |
| `src/config/base.py`, `quantize.py`, `benchmark.py` | **Empty** |
| `src/data/calibration.py`, `dataloader.py` | Untracked, content unknown |
| `serve.py` | Untracked |

Critical blocker: every quantizer imports `QuantizeConfig` from `src.config.schemas` but schema file is empty. Must restore schemas first.

## 3. Hardware / Precision Compatibility Matrix

| Precision | Hardware floor | Inference runtime | AWQ-compatible? |
|---|---|---|---|
| INT8 W8A8 | Any CUDA ≥ Turing | vLLM, TensorRT-LLM | No (use GPTQ/SmoothQuant) |
| INT4 W4A16 | Any CUDA ≥ Ampere | vLLM Marlin, ExLlamaV2 | **Yes** |
| INT4 W4A8 | Hopper preferred | vLLM Marlin-FP8 | No (mixed) |
| FP8 E4M3/E5M2 | Hopper (H100, H200), Ada (L40S, RTX 6000 Ada) | vLLM, TensorRT-LLM | No |
| FP8 BLOCK | Hopper+ | vLLM | No |
| NVFP4 | Blackwell (B100, B200, RTX 50) | vLLM ≥ 0.7, TRT-LLM | No (PTQ only) |
| MXFP4 | Blackwell native; Hopper emulated | vLLM (emulated/native) | No (PTQ only) |

**Implication:** AWQ class supports only `W4A16` / `W4A16_ASYM`. All sub-byte float schemes (FP8, NVFP4, MXFP4) go through a separate PTQ path. Mixing them silently is the failure mode to prevent.

## 3b. Backend Capability Matrix

| Capability | llmcompressor | modelopt |
|---|---|---|
| INT8 W8A8 / W8A16 | ✅ | ✅ |
| INT4 W4A16 (AWQ) | ✅ (`AWQModifier`) | ✅ (`INT4_AWQ_CFG`) |
| INT4 W4A8 | ✅ | ✅ |
| GPTQ | ✅ (`GPTQModifier`) | ⚠️ (limited; AWQ preferred) |
| SmoothQuant | ✅ | ✅ |
| FP8 per-tensor / dynamic | ✅ | ✅ (`FP8_DEFAULT_CFG`) |
| FP8 per-channel/per-token | ✅ | ✅ (`FP8_PER_CHANNEL_PER_TOKEN_CFG`) |
| FP8 block (`FP8_PB_WO`) | ⚠️ partial | ✅ |
| NVFP4 | ❌ | ✅ (`NVFP4_DEFAULT_CFG`, `NVFP4_MLP_ONLY_CFG`, `NVFP4_MOE_ONLY_CFG`) |
| MXFP4 | ❌ | ✅ (`MXFP4_DEFAULT_CFG`, native GPT-OSS, NVFP4 convertible) |
| MXFP8 | ❌ | ✅ |
| SVDQuant (low-rank + 4-bit) | ❌ | ✅ |
| Double Quantization | ❌ | ✅ |
| AutoQuantize (per-layer search) | ❌ | ✅ (`mtq.auto_quantize`) |
| KV-cache quant (FP8 / NVFP4) | ⚠️ via runtime | ✅ first-class |
| Minitron pruning (depth + width) | ❌ | ✅ (`mtp.prune`) |
| Knowledge distillation | ❌ | ✅ (`mtd.convert`) |
| 2:4 sparsity | ❌ | ✅ (`mts.sparsify`) |
| Export to vLLM | ✅ native | ✅ via `export_hf_checkpoint` |
| Export to TensorRT-LLM | ❌ | ✅ native |
| Export to SGLang | ✅ | ✅ |
| Mamba pruning | ❌ | ✅ (`mamba_num_heads`, `mamba_head_dim`) |
| MoE expert pruning | ❌ | ✅ (`num_moe_experts`, `moe_ffn_hidden_size`) |

**Decision rule:** scheme + method routes to backend automatically.
- INT4 AWQ / INT4 GPTQ / INT8 / standard FP8 → `llmcompressor`
- NVFP4 / MXFP4 / MXFP8 / SVDQuant / AutoQuantize / KV-cache quant / pruning / distillation → `modelopt`
- User override via `backend: llmcompressor | modelopt | auto` (default `auto`).

## 4. Target Architecture

```
src/
  config/
    schemas.py          # QuantizeConfig / PruneConfig / DistillConfig
    quantize.py         # YAML loader → typed config
    base.py             # Pydantic base + shared validators
  quantization/
    __init__.py         # Re-export get_quantizer, list_methods
    registry.py         # (existing) decorator registry
    base.py             # (refactor) BaseQuantizer — model load + save only
    schemes.py          # NEW — QuantScheme enum + backend dispatch
    arch_profiles.py    # NEW — per-architecture mappings + ignore patterns
    backends/
      __init__.py       # backend registry (llmcompressor, modelopt)
      llmcompressor.py  # NEW — adapter: schemes → llmcompressor modifiers
      modelopt.py       # NEW — adapter: schemes → mtq configs + export
    awq.py              # (refactor) INT4-only, picks backend
    gptq.py             # NEW — INT4 / W4A8 / FP8-weight
    ptq.py              # NEW — FP8 / NVFP4 / MXFP4 / MXFP8 / INT8
    autoquant.py        # NEW — modelopt AutoQuantize per-layer search
    svdquant.py         # NEW — modelopt SVDQuant (low-rank + 4-bit)
    smoothquant.py      # NEW — optional pre-pass for INT8/FP8 activation
    fp16.py             # NEW — no-op baseline
    kv_cache.py         # NEW — KV cache FP8/NVFP4 (modelopt only)
  pruning/
    __init__.py
    base.py             # NEW — BasePruner abstract
    minitron.py         # NEW — modelopt mtp.prune wrapper (depth + width)
    importance.py       # NEW — activation/Taylor importance scoring
    fastnas.py          # NEW (later) — modelopt FastNAS for CV
  distillation/
    __init__.py
    base.py             # NEW — teacher/student wiring
    minitron_recipe.py  # NEW — post-prune distill loop (80-100B tokens)
    losses.py           # NEW — KD losses (logit KL, hidden MSE, attn match)
  sparsity/             # (Phase 9+, optional)
    __init__.py
    semi_structured.py  # 2:4 sparsity via modelopt mts.sparsify
  data/
    calibration.py      # Dataset → token batches, modality-aware
    dataloader.py       # HF datasets wrapper
    distill_data.py     # NEW — long-horizon distill dataset streaming
  export/
    __init__.py
    hf_export.py        # modelopt export_hf_checkpoint wrapper
    trtllm_export.py    # legacy TRT-LLM export
```

## 5. Component Specifications

### 5.1 `config/schemas.py`

Pydantic v2 models. Single source of truth.

- `ModelConfig`: `model_id`, `torch_dtype`, `device_map`, `trust_remote_code`, `min_pixels`, `max_pixels`, `modality` (`auto` | `text` | `vision` | `audio`).
- `SchemeConfig`: `scheme` (Literal of all supported names from `QuantScheme`), `group_size` (default 128), `symmetric` (bool), `actorder` (bool), `block_size` (for FP8_BLOCK), `targets` (Linear by default), `ignore` (override list).
- `CalibrationConfig`: `dataset_name`, `num_samples`, `max_seq_len`, `image_field`, `text_field`, `seed`.
- `OutputConfig`: `output_dir`, `save_compressed` (bool), `save_processor`, `push_to_hub`.
- `QuantizeConfig`: top-level — `method` (str, matches registry key), `model: ModelConfig`, `scheme: SchemeConfig`, `calibration: CalibrationConfig`, `output: OutputConfig`, `smoothquant: SmoothQuantConfig | None`.

Validator: reject `method="awq"` with `scheme` not in `{W4A16, W4A16_ASYM}`. Reject FP8/NVFP4/MXFP4 with `method="awq"`.

### 5.2 `quantization/schemes.py`

```
QuantScheme = StrEnum:
  W8A8_INT, W8A16_INT,
  W4A16_INT, W4A16_INT_ASYM, W4A8_INT, W4A4_INT,
  FP8, FP8_DYNAMIC, FP8_BLOCK,
  NVFP4, NVFP4A16,
  MXFP4, MXFP4A16
```

`build_quantization_modifier(scheme, targets, ignore, group_size, ...) → QuantizationModifier`:
- Maps each enum → preset string accepted by llmcompressor (`"W4A16"`, `"FP8_DYNAMIC"`, `"NVFP4"`, `"MXFP4A16"`, etc.).
- Versions of llmcompressor differ on enum names — wrap in a small adapter that probes `llmcompressor.modifiers.quantization` preset registry at import time and raises a clear error if a scheme is unsupported by the installed version.

`requires_calibration(scheme) → bool`: True for INT4/INT8/NVFP4 weight schemes and any A8/A4 activation; False for FP8_DYNAMIC and FP8 weight-only.

`hardware_floor(scheme) → str`: returns capability tag (`sm_70`, `sm_80`, `sm_89`, `sm_90`, `sm_100`) for runtime warnings.

### 5.3 `quantization/arch_profiles.py`

`ArchProfile` dataclass:
- `family`: `dense_llm | vlm | moe | mamba | hybrid`
- `awq_mappings`: `list[AWQMapping]` — required only for AWQ method
- `ignore_patterns`: `list[str]` — regex/glob of modules to skip
- `calibration_modality`: `text | vision_text | audio_text`
- `sequential_targets`: `list[str] | None` — layer types for sequential quant (memory saving on big MoE)
- `requires_cpu_offload`: bool

`detect_profile(model) → ArchProfile`:
1. Read `model.config.architectures[0]` (e.g., `Qwen2_5_VLForConditionalGeneration`, `MixtralForCausalLM`, `MambaForCausalLM`, `JambaForCausalLM`).
2. Read `model.config.model_type`.
3. Dispatch to family-specific builder.

Family builders:

**`_profile_dense_llm(model)`** — Llama, Qwen2/Qwen2.5, Mistral, Gemma, Phi:
- Ignore: `lm_head`
- AWQ mappings: standard (`[".*q_proj", ".*k_proj", ".*v_proj"]` smoothed by `re:.*input_layernorm$`; `[".*gate_proj", ".*up_proj"]` by `re:.*post_attention_layernorm$`; `[".*down_proj"]` by `re:.*up_proj$`).

**`_profile_vlm(model)`** — Qwen2-VL, Qwen2.5-VL, LLaVA, InternVL, Pixtral:
- Ignore: `lm_head`, `re:.*visual\..*`, `re:.*vision_tower\..*`, `re:.*merger\..*`, `re:.*mm_projector.*`.
- AWQ mappings: same as dense but scoped under `language_model.*` prefix when language tower is nested.
- `calibration_modality = vision_text`.

**`_profile_moe(model)`** — Mixtral, Qwen2-MoE, DeepSeek-V2/V3, OLMoE, Phi-MoE:
- Ignore: `lm_head`, `re:.*gate$`, `re:.*router\..*`, `re:.*shared_expert_gate$`.
- AWQ mappings: per-expert variant — `re:.*experts\.\d+\.w1`, `.w2`, `.w3` (Mixtral); `re:.*experts\.\d+\.(gate_proj|up_proj|down_proj)` (Qwen-MoE).
- `sequential_targets = ["MixtralDecoderLayer"]` (or arch-specific) to quantize layer-by-layer.
- `requires_cpu_offload = True` for ≥ 70B-class MoE.

**`_profile_mamba(model)`** — Mamba, Mamba2, Falcon-Mamba:
- Ignore (hard requirement): `re:.*conv1d.*`, `re:.*dt_proj.*`, `re:.*A_log.*`, `re:.*D$`, `re:.*x_proj.*`. Quantizing SSM state-space params destroys recurrence.
- Quantizable: `in_proj`, `out_proj`, embeddings (optional).
- **AWQ unsuitable** — activation-aware scaling is calibrated for attention/MLP, not SSM. `_profile_mamba` raises `UnsupportedMethodError` if method=AWQ. PTQ FP8 or INT8 weight-only is the recommended path.

**`_profile_hybrid(model)`** — Jamba, Zamba, Zamba2 (interleaved attention + Mamba):
- Union of dense_llm mappings (for attention blocks) and Mamba ignore (for SSM blocks).
- AWQ: only applied to attention layers via `targets` list scoped to `re:.*self_attn.*` and `re:.*mlp\.(gate_proj|up_proj|down_proj)`.

Extensibility: profile builders registered via decorator `@register_profile("Qwen2_5_VLForConditionalGeneration")` so adding new architectures = one function, no core edits.

### 5.4 `quantization/base.py` (refactor)

Keep `load_model` / `save`. Add:
- `self.profile: ArchProfile | None = None` — populated in `load_model` via `detect_profile`.
- Modality-aware loader: profile decides `AutoModelForCausalLM` vs `AutoModelForImageTextToText` vs `AutoModelForVision2Seq` rather than the current string-match on `"VL"`.
- `_apply_smoothquant(recipe_list)` helper — prepends `SmoothQuantModifier` if config requests it (needed for accurate INT8 / W8A8 / FP8 activations on activation-heavy archs).
- `_validate_scheme_supported(scheme)` — checks profile.family vs scheme using a compatibility table; raises clear error before wasting calibration time.

### 5.5 `quantization/awq.py` (refactor)

- Constructor: assert `config.scheme.scheme in {W4A16_INT, W4A16_INT_ASYM}` — fail fast.
- `quantize(dataset, output_dir)`:
  1. Build recipe: `[AWQModifier(mappings=profile.awq_mappings, ignore=profile.ignore_patterns), QuantizationModifier(...)]` from `schemes.build_quantization_modifier`.
  2. Call `oneshot(model=self.model, dataset=dataset, recipe=recipe, max_seq_length=..., num_calibration_samples=..., sequential_targets=profile.sequential_targets)`.
  3. Delegate save to `BaseQuantizer.save`.
- Drop the hardcoded `_detect_num_layers` — now lives in `ArchProfile` if needed.
- Drop Qwen2.5-VL-specific branching.

### 5.6 `quantization/gptq.py` (new)

- Wraps `GPTQModifier`. Accepts schemes: `W4A16_INT`, `W4A16_INT_ASYM`, `W8A16_INT`, `W4A8_INT`, `FP8` (weight-only).
- Same profile flow as AWQ; no `AWQMapping` needed.
- Better than AWQ for very deep / very wide models where AWQ scale search diverges (>70B dense, large MoE).

### 5.7 `quantization/ptq.py` (new)

- For schemes where no calibration-driven scale search is needed or where AWQ/GPTQ don't apply.
- Single-modifier recipe: `[QuantizationModifier(scheme=...)]`.
- Required for: `FP8`, `FP8_DYNAMIC`, `FP8_BLOCK`, `NVFP4`, `NVFP4A16`, `MXFP4`, `MXFP4A16`, `W8A8_INT` (optionally preceded by SmoothQuant).
- `FP8_DYNAMIC` is data-free → calibration dataset optional → fastest path.

### 5.8 `quantization/smoothquant.py` (new)

- Wraps `SmoothQuantModifier(smoothing_strength=0.8, mappings=profile.smoothquant_mappings)`.
- Engaged when `config.smoothquant is not None` and scheme has activation quant (W8A8, W4A8, FP8 act).
- Profile carries `smoothquant_mappings` separately (LayerNorm-aware).

### 5.9 `quantization/fp16.py` (new)

- Identity quantizer. Loads model, saves at FP16/BF16, no modifier. Lets benchmark code compare against an un-quantized baseline through the same interface.

### 5.10a `quantization/backends/modelopt.py` (new)

Thin adapter wrapping `modelopt.torch.quantization` (`mtq`).

Entry points:
- `mtq.quantize(model, config, forward_loop)` — main PTQ call. `config` is a dict, e.g.:
  - `mtq.NVFP4_DEFAULT_CFG`
  - `mtq.NVFP4_MLP_ONLY_CFG` (NVFP4 on MLP only, attention untouched — speeds calib + preserves accuracy)
  - `mtq.NVFP4_MOE_ONLY_CFG` (NVFP4 on MoE experts only)
  - `mtq.MXFP4_DEFAULT_CFG`
  - `mtq.MXFP8_DEFAULT_CFG`
  - `mtq.FP8_DEFAULT_CFG` / `mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG` / `mtq.FP8_PB_WO_CFG`
  - `mtq.INT8_DEFAULT_CFG` / `mtq.INT8_SMOOTHQUANT_CFG`
  - `mtq.INT4_AWQ_CFG` / `mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG`
  - `mtq.W4A8_AWQ_BETA_CFG`
- `forward_loop(model)` — calibration callable that runs `num_calibration_samples` batches through `model`. Built from `data/calibration.py`.
- `modelopt.torch.export.export_hf_checkpoint(model, export_dir)` — produces vLLM/TRT-LLM-loadable checkpoint with `hf_quant_config.json`.

Wrap each modelopt config in our `QuantScheme` enum so caller never references `mtq.*` directly. Probe `modelopt` version at import; raise clean error if missing (it's a separate `pip install nvidia-modelopt[torch]`).

KV-cache quant: `mtq.config` accepts `*_kv_cache` flags — exposed via `SchemeConfig.kv_cache_dtype: "fp8" | "nvfp4" | None`.

### 5.10b `quantization/autoquant.py` (new, modelopt-only)

Wraps `mtq.auto_quantize(model, search_config, calib_fn, constraints={"effective_bits": 4.5})`. Per-layer precision search: each linear layer can independently land at INT4 / FP8 / NVFP4 / FP16 based on sensitivity vs target effective bitwidth. Output is heterogeneous-precision checkpoint.

Use when: accuracy of uniform W4A16 drops too much but uniform W8A8 wastes budget. Computationally heavier — gradient + Hessian-style sensitivity per layer.

### 5.10c `quantization/svdquant.py` (new, modelopt-only)

Wraps `mtq.svdquant` (low-rank residual + 4-bit base). Reduces 4-bit quantization error by absorbing high-magnitude outliers into a small low-rank SVD adapter. Cost: extra inference latency (the rank-r matmul); benefit: closes the AWQ accuracy gap for harder models.

### 5.10d `quantization/kv_cache.py` (new, modelopt-only)

KV cache PTQ. `mtq` configs accept `*_KV_CFG` variants (e.g., `FP8_KV_CFG`, `NVFP4_KV_CFG`) — at decode time the K/V tensors are stored in low precision, cutting memory by 2× (FP8) or 4× (NVFP4). Independent of weight scheme; can combine FP16 weights + FP8 KV cache.

### 5.11 `pruning/minitron.py` (new)

Wraps `modelopt.torch.prune.mtp.prune`.

`MinitronPruner(config: PruneConfig)`:
- `mode`: `depth` | `width` | `depth_width`
- `targets`:
  - depth: `num_layers` (e.g., 32 → 24)
  - width: `hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `num_kv_heads`
  - Mamba: `mamba_num_heads`, `mamba_head_dim`
  - MoE: `num_moe_experts`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`
- `importance_scoring`: activation-magnitude (forward hooks over calibration data) — `mtp` handles internally, but accept `num_samples` (default 1024) and `seq_len` (default 8192).
- `restore_after_prune: bool` — save heterogeneous structure via `mto.save(model, path)` / `mto.restore(model, path)` so loaded model matches new dims.

Flow:
1. Load teacher (full) model.
2. Run `mtp.prune(model, mode=..., constraints={...}, forward_loop=calib_loop)`.
3. Save pruned student via `mto.save`.
4. Hand off to distillation phase.

Hard constraint: Minitron requires NeMo/Megatron-core for some archs. HF-only path supported for Llama, Qwen, Mistral, Mixtral, Mamba. Verify per-arch via small wrapper that maps HF config → modelopt-supported architecture name; raise `UnsupportedArchitectureError` cleanly.

### 5.12 `distillation/minitron_recipe.py` (new)

Post-prune accuracy recovery. Without this, depth-pruning loses ~5–15 PPL.

`MinitronDistiller(config: DistillConfig)`:
- Wraps `modelopt.torch.distill.mtd.convert(student, teacher, distillation_config)`.
- Loss: composite (logit KL + hidden-state MSE + attention map MSE) — exposed via `losses.py`.
- Schedule:
  - Tokens: 80–100B (paper recommendation) — bench machines won't survive this; expose `tokens_seen` param, default 1B for smoke, document the gap.
  - LR: 1e-4 warmup → cosine → 1e-6.
  - Batch size: 768 (16 × 48 GPUs in paper); locally tune by VRAM.
- Data: a long-horizon stream (`distill_data.py`) — FineWeb / C4 / domain mix per `DistillConfig.dataset`.

Output: distilled checkpoint that goes back into the quantization pipeline (e.g., Minitron-prune → distill → AWQ-INT4).

### 5.13 `pruning/fastnas.py` (later)

`mtp.prune(model, mode="fastnas", ...)` for CV (ViT, ConvNets). Out of scope for Phase 6; placeholder.

### 5.14 `sparsity/semi_structured.py` (Phase 9+, optional)

`mts.sparsify(model, mode="sparse_2_4", forward_loop=...)`. 2:4 sparsity on Ampere+. Composes with quantization (e.g., 2:4 sparse + FP8). Skipped until base pipeline stable.

### 5.10 `data/calibration.py`

- `build_calibration_dataset(config: CalibrationConfig, profile: ArchProfile, processor_or_tokenizer)`:
  - Text-only: stream `wikitext` / `c4` / user-specified, truncate to `max_seq_len`.
  - Vision-text: `flickr30k` / `coco` / user-specified — apply `processor` to (image, prompt) pairs, return dicts with `pixel_values` + `input_ids`.
  - Audio-text (future): placeholder.
- Sample count default: 512 (AWQ), 256 (GPTQ), 128 (PTQ INT8), 0 (FP8_DYNAMIC).

## 6. Config Surface (example, not code)

`configs/quantize/qwen25vl_3b_awq.yaml` — INT4 W4A16, AWQ, backend=llmcompressor.
`configs/quantize/qwen25vl_3b_fp8.yaml` — FP8_DYNAMIC, PTQ, backend=llmcompressor.
`configs/quantize/qwen25vl_3b_nvfp4.yaml` — NVFP4_MLP_ONLY, PTQ, backend=modelopt, requires Blackwell.
`configs/quantize/qwen25vl_3b_mxfp4.yaml` — MXFP4_DEFAULT, backend=modelopt.
`configs/quantize/qwen25vl_3b_autoq.yaml` — AutoQuantize, effective_bits=4.5, backend=modelopt.
`configs/quantize/qwen25vl_3b_svdq.yaml` — SVDQuant rank=32, backend=modelopt.
`configs/quantize/qwen25vl_3b_fp8_kv.yaml` — FP16 weights + FP8 KV cache, backend=modelopt.
`configs/quantize/mixtral_8x7b_gptq_w4a16.yaml` — GPTQ INT4 with sequential_targets.
`configs/quantize/mixtral_8x7b_nvfp4_moe.yaml` — NVFP4_MOE_ONLY, backend=modelopt.
`configs/quantize/mamba_2_8b_fp8.yaml` — PTQ FP8 weight-only, Mamba ignore list active.
`configs/quantize/jamba_52b_w4a16.yaml` — GPTQ INT4 on attention only, SSM blocks skipped.
`configs/prune/llama3_8b_to_4b_width.yaml` — Minitron width prune `hidden_size 4096→3072`, `ffn 14336→9216`, `num_heads 32→24`.
`configs/prune/llama3_8b_to_4b_depth.yaml` — Minitron depth prune `num_layers 32→16`.
`configs/prune/mixtral_experts_8_to_4.yaml` — MoE expert prune `num_moe_experts 8→4`.
`configs/prune/mamba_2_8b_heads.yaml` — Mamba prune `mamba_num_heads`, `mamba_head_dim`.
`configs/distill/llama3_8b_to_4b.yaml` — distillation recipe (teacher=8B, student=pruned 4B, 1B–100B tokens).
`configs/pipeline/llama3_8b_prune_distill_quant.yaml` — full chain: prune → distill → NVFP4 quant.

## 7. Failure Modes Handled Up Front

| Failure | Guard |
|---|---|
| AWQ + FP8 silently degrading to RTN | SchemeConfig validator rejects at config-load time |
| Mamba SSM quantized → garbage output | Mamba profile `ignore` list + AWQ refusal |
| MoE OOM during oneshot | `sequential_targets` + optional `requires_cpu_offload` |
| NVFP4 on non-Blackwell | `hardware_floor` check at start of `quantize()`, hard error |
| VLM vision tower quantized → broken image features | VLM profile `ignore` list covers `visual.*`, `vision_tower.*`, `merger.*` |
| llmcompressor version doesn't ship scheme preset | `schemes.py` probes presets at import, fails with version hint |
| `save_pretrained` KeyError on VLM `module_map` | Preserved from current `base.py` (CPU move + `del hf_device_map`) |
| modelopt not installed but NVFP4/MXFP4 requested | Backend dispatcher detects missing `modelopt` import, raises with `pip install nvidia-modelopt[torch]` hint |
| modelopt checkpoint loaded by vLLM with wrong `quantization=` flag | `export/hf_export.py` writes `hf_quant_config.json` + sidecar `LOAD_HINT.md` documenting flag (`modelopt` / `modelopt_fp4` / `modelopt_mxfp8`) |
| Pruning without distillation → PPL spike | `PruneConfig.distill_after: bool = True` default; warn loudly if False |
| Minitron prune on unsupported HF arch | Arch whitelist in `pruning/minitron.py`; cleanly errors out, suggests NeMo-container path |
| AutoQuantize blowing wallclock budget | `autoquant.py` accepts `max_search_hours`; falls back to uniform scheme on timeout |
| SVDQuant inference slower than pure INT4 | Document trade-off; benchmark adds latency column for SVDQuant variants |
| KV-cache quant + paged attention mismatch in vLLM | Pin vLLM version per scheme in `configs/`; document tested matrix |

## 8. Phased Implementation Order

1. **Phase 0 — restore foundations.** Rebuild `src/config/schemas.py`, `config/base.py`, `config/quantize.py`. Add `PruneConfig`, `DistillConfig`. Unblocks everything else.
2. **Phase 1 — schemes + profiles + backend registry.** Write `schemes.py`, `arch_profiles.py`, `backends/llmcompressor.py`, `backends/modelopt.py` (skeleton). Import smoke test on both backends.
3. **Phase 2 — refactor AWQ (llmcompressor).** Move Qwen2.5-VL specifics out, switch to profile lookup. Verify Qwen2.5-VL-3B INT4.
4. **Phase 3 — PTQ + FP16 (llmcompressor).** `ptq.py` FP8_DYNAMIC + `fp16.py` baseline. Reproduce Qwen2.5-VL FP8 end-to-end.
5. **Phase 4 — GPTQ + SmoothQuant (llmcompressor).** Validate Llama-3-8B INT4 + Llama-3-8B INT8 SmoothQuant.
6. **Phase 5 — MoE profile.** Mixtral-8x7B GPTQ W4A16 sequential quant.
7. **Phase 6 — Mamba + hybrid.** Mamba-2.8B FP8 PTQ; Jamba attention-only INT4.
8. **Phase 7 — modelopt quantization path.** `backends/modelopt.py` full impl + `ptq.py` route. Validate FP8 via modelopt path against llmcompressor (parity check). Add `kv_cache.py` FP8 KV.
9. **Phase 8 — NVFP4 / MXFP4 / MXFP8.** Gated by hardware check. NVFP4_MLP_ONLY and NVFP4_MOE_ONLY variants. Export via `export_hf_checkpoint`. Verify vLLM load with `quantization="modelopt_fp4"`.
10. **Phase 9 — AutoQuantize + SVDQuant.** Per-layer precision search on Qwen2.5-VL-7B (effective_bits=4.5). SVDQuant rank-32 on Llama-3-8B.
11. **Phase 10 — Pruning (Minitron, depth).** `pruning/minitron.py` depth-prune Llama-3-8B → 4B. `mto.save` heterogeneous checkpoint.
12. **Phase 11 — Pruning (width + MoE + Mamba).** Width prune Llama-3-8B. Expert prune Mixtral 8→4. Mamba head prune.
13. **Phase 12 — Distillation pipeline.** `distillation/minitron_recipe.py`. Run 1B-token smoke distill on Llama-3-8B→4B. Document full 80–100B as documented-but-not-executed (compute budget).
14. **Phase 13 — Full pipeline.** Chain prune → distill → quant. Reference recipe: Llama-3-8B → 4B (prune) → distill 1B tokens → NVFP4 (modelopt).
15. **Phase 14 — Calibration dataset + configs.** Rebuild deleted YAMLs against new schema. Add `distill_data.py` long-horizon streamer.
16. **Phase 15 — Sparsity (optional).** 2:4 sparsity via `mts.sparsify`. Compose with FP8.
17. **Phase 16 — Evaluation harness.** Accuracy (MMLU, HumanEval, MMMU for VLM), latency (vLLM bench), memory (peak alloc), perplexity (wikitext). Cross-backend comparison tables.

## 9. Open Questions (need user input before Phase 0)

1. Backend stance — agreed dual-backend (`llmcompressor` + `modelopt`)? Or modelopt-only (drops Phase 2–6 in favor of modelopt-equivalent), or llmcompressor-only (drops NVFP4/MXFP4/pruning/distillation)?
2. Target hardware — benchmark machine GPU? (Determines: Hopper → FP8 path; Blackwell → NVFP4/MXFP4 path; Ampere only → INT4/INT8 path; CPU/MPS → emulated only.)
3. Compute budget for distillation — paper recipe is 80–100B tokens; if budget is sub-1B, document as scaffolded-but-not-validated.
4. NeMo container acceptable? Some Minitron paths (Megatron-core models) need `nvcr.io/nvidia/nemo:26.04`. If HF-only, restrict to whitelist (Llama, Qwen, Mistral, Mixtral, Mamba).
5. Keep `src/integrations/` (untracked) — what's in it? May overlap with `backends/` or `export/`.
6. Pydantic v1 or v2 for schemas — affects validator syntax.
7. Should `fp16.py` run a forward pass to warm caches, or pure no-op pass-through?
8. AutoQuantize search budget — wallclock cap per model? (Search is gradient/Hessian-heavy, multi-hour on 8B+.)
9. Sparsity scope — include 2:4 in Phase 15, or defer indefinitely?
10. Export targets — vLLM only, or also TRT-LLM + SGLang? Affects `export/` scope.

## 10. References

- NVIDIA TensorRT Model Optimizer — main repo / docs / changelog
- NVIDIA Model Optimizer PTQ examples — `examples/llm_ptq/README.md`
- NVIDIA Model Optimizer pruning examples — `examples/pruning/README.md`
- NVFP4 blog — "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"
- Minitron paper — "LLM Pruning and Distillation in Practice: The Minitron Approach" (arXiv 2408.11796)
- vLLM ModelOpt integration docs — `docs.vllm.ai/en/stable/features/quantization/modelopt/`
- llmcompressor repo + `compressed-tensors`
- TensorRT-LLM quantization docs
- NeMo pruning/distillation tutorial — `NeMo/tutorials/llm/qwen/pruning-distillation`