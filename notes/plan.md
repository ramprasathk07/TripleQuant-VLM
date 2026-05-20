# Unified Quantization Layer — Implementation Plan

## 1. Goal

Refactor `src/quantization/` into a model-agnostic, precision-agnostic compression layer that handles:

- **Model families:** dense LLM, VLM (vision-language), MoE (mixture-of-experts), Mamba / state-space, hybrid (Jamba, Zamba, Falcon-Mamba).
- **Methods:** AWQ (INT4-only by definition), GPTQ, PTQ / RTN, SmoothQuant pre-pass.
- **Precisions:** INT8 (W8A8 / W8A16), INT4 (W4A16, W4A8, W4A4), FP8 (E4M3, E5M2, dynamic, block), NVFP4 (Blackwell), MXFP4 (OCP microscaling).

Backend: `llmcompressor` (oneshot pipeline) + `compressed-tensors` serialization. vLLM consumes output directly.

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

## 4. Target Architecture

```
src/
  config/
    schemas.py          # QuantizeConfig + nested model/scheme/calibration configs
    quantize.py         # YAML loader → QuantizeConfig
    base.py             # Pydantic base + shared validators
  quantization/
    __init__.py         # Re-export get_quantizer, list_methods
    registry.py         # (existing) decorator registry
    base.py             # (refactor) BaseQuantizer — model load + save only
    schemes.py          # NEW — QuantScheme enum + modifier factory
    arch_profiles.py    # NEW — per-architecture mappings + ignore patterns
    awq.py              # (refactor) INT4-only, uses ArchProfile
    gptq.py             # NEW — INT4 / W4A8 / FP8-weight via GPTQModifier
    ptq.py              # NEW — FP8 / NVFP4 / MXFP4 / INT8 via QuantizationModifier
    smoothquant.py      # NEW — optional pre-pass for INT8/FP8 activation quant
    fp16.py             # NEW — no-op baseline (passthrough for benchmark)
  data/
    calibration.py      # Dataset → token batches, modality-aware
    dataloader.py       # HF datasets wrapper
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

### 5.10 `data/calibration.py`

- `build_calibration_dataset(config: CalibrationConfig, profile: ArchProfile, processor_or_tokenizer)`:
  - Text-only: stream `wikitext` / `c4` / user-specified, truncate to `max_seq_len`.
  - Vision-text: `flickr30k` / `coco` / user-specified — apply `processor` to (image, prompt) pairs, return dicts with `pixel_values` + `input_ids`.
  - Audio-text (future): placeholder.
- Sample count default: 512 (AWQ), 256 (GPTQ), 128 (PTQ INT8), 0 (FP8_DYNAMIC).

## 6. Config Surface (example, not code)

`configs/quantize/qwen25vl_3b_awq.yaml` — INT4 W4A16, AWQ method.
`configs/quantize/qwen25vl_3b_fp8.yaml` — FP8_DYNAMIC, PTQ method.
`configs/quantize/qwen25vl_3b_nvfp4.yaml` — NVFP4, PTQ method, requires Blackwell.
`configs/quantize/mixtral_8x7b_gptq_w4a16.yaml` — GPTQ INT4 with `sequential_targets`.
`configs/quantize/mamba_2_8b_fp8.yaml` — PTQ FP8 weight-only, Mamba ignore list active.
`configs/quantize/jamba_52b_w4a16.yaml` — GPTQ INT4 on attention only, SSM blocks skipped.

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

## 8. Phased Implementation Order

1. **Phase 0 — restore foundations.** Rebuild `src/config/schemas.py`, `config/base.py`, `config/quantize.py`. Unblocks everything else.
2. **Phase 1 — schemes + profiles.** Write `schemes.py` and `arch_profiles.py` with dense_llm + vlm profiles. Run import smoke test.
3. **Phase 2 — refactor AWQ.** Move Qwen2.5-VL specifics out, switch to profile lookup. Verify against existing Qwen2.5-VL-3B INT4 run.
4. **Phase 3 — add PTQ + FP16.** `ptq.py` for FP8_DYNAMIC (data-free, fastest to validate). `fp16.py` baseline. Reproduce a Qwen2.5-VL FP8 model end-to-end.
5. **Phase 4 — add GPTQ + SmoothQuant.** Validate on a dense LLM (Llama-3-8B INT4).
6. **Phase 5 — MoE profile.** Mixtral-8x7B GPTQ W4A16 with sequential quant.
7. **Phase 6 — Mamba + hybrid profile.** Mamba-2.8B FP8 PTQ; Jamba attention-only INT4.
8. **Phase 7 — NVFP4 / MXFP4.** Gated behind hardware check; validate on rented Blackwell or via vLLM emulation.
9. **Phase 8 — calibration dataset module + configs.** Rebuild deleted YAMLs against new schema.
10. **Phase 9 — restore evaluation/** (accuracy, latency, memory, perplexity) — out of scope of this plan but unblocked once quantizers stabilize.

## 9. Open Questions (need user input before Phase 0)

1. Confirm `llmcompressor` is the only backend, or also wire `llm-awq` (per commit `36c6f55`) and `auto-gptq`/`gptqmodel` as alternate backends behind a `backend:` field.
2. Target hardware mix — does benchmark machine have Hopper (FP8) or Blackwell (NVFP4/MXFP4)? Determines which Phase 7 scope is reachable.
3. Keep `src/integrations/` (untracked dir) — what does it contain? May overlap with planned `schemes.py`.
4. Pydantic v1 or v2 for schemas — affects validator syntax.
5. Should `fp16.py` actually run a forward pass to warm caches, or be a pure no-op pass-through?