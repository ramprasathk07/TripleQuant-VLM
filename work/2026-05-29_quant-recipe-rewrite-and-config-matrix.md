# 2026-05-29 — llm_compressor recipe rewrite + full quant/benchmark config matrix

**Branch:** main · **Status:** uncommitted (working tree)

Summary of everything changed today. Two threads: (1) make the
`llm_compressor` quantization path feasible for *all* formats it supports +
clean it up; (2) wipe the old quant configs and rebuild a comprehensive test
matrix for three models with matching benchmark configs.

---

## 1. Formatting cleanup — `src/quantization/llm_compressor.py`

- Deleted ~120-line commented-out dead `_build_recipe` block.
- Tightened AI-slop vertical spacing in the live builder.
- No behavior change in this step (cleanup only).

## 2. Feasibility fix — preset-driven recipe (the core change)

**Problem found:** the old `_build_recipe` hardcoded `"type": "int"` and only
parsed `W4`/`W8` prefixes → every float scheme crashed:

| Format | Was | Now |
|---|---|---|
| W4A16 / W4A16_ASYM / W8A16 / W8A8 (int) | ✅ | ✅ |
| FP8 static | ❌ `ValueError("Unsupported scheme")` | ✅ |
| FP8_DYNAMIC | ❌ | ✅ |
| NVFP4 / NVFP4A16 | ❌ | ✅ |
| W4A8 | ❌ not in enum | ✅ |
| plain PTQ (no algo) | ❌ unreachable (`MethodLiteral` had no `ptq`) | ✅ |
| `method: smoothquant` | ❌ Pydantic reject | fixed configs use `method: ptq` |
| MXFP4 / FP8_BLOCK | ❌ | correctly rejected → "use backend=modelopt" |

**Fix:** new `_build_quant_config()` resolves
`compressed_tensors.quantization.preset_name_to_scheme(scheme, targets)` as the
source of truth (correct int/float dtype, strategy, group_size, dynamic for any
scheme), then overlays user knobs *only where safe*:
- `group_size` override only on `group` strategy.
- `per_channel` → forces `channel` strategy (int only), drops group_size.
- `symmetric`/`observer` overrides INT-only (float schemes keep preset).
- `actorder` GPTQ-only AND group-strategy-only (channel/tensor reject it).

`config_groups` entries now carry their own `targets` (required by the modifier
schema in llmcompressor 0.6.0 — the old "targets must not be inside" comment was
wrong for this version).

**Verified against installed stack** (llmcompressor 0.6.0 / compressed-tensors
0.10). Preset schemes available: `W4A16, W4A16_ASYM, W4A8, W8A16, W8A8, INT8,
FP8, FP8_DYNAMIC, NVFP4, NVFP4A16`.

## 3. Schema fixes — `src/config/schemas.py`

- `MethodLiteral` += `"ptq"`.
- `QuantScheme`/`QuantSchemeLiteral` += `W4A8`, `NVFP4A16`.
- `ModalityLiteral` += `"vision_text"` (every VLM quant config was failing to
  load on this value; `modality` is descriptive-only, unused in `src/`).
- `extra="forbid"` on `SchemeConfig`, `AWQParams`, `GPTQParams`,
  `SmoothQuantConfig` → silently-dropped config keys now error loudly.
- `GPTQParams` += `block_size` (real GPTQModifier field).
- Extended `_validate_method_scheme_compat` (llm_compressor backend only):
  awq→W4A16 set, gptq→int-set, float→ptq-only, modelopt-only→reject.

## 4. Real bugs this surfaced (silent before)

- AWQ `search_steps` / `clipping` / `percentile` config keys did **nothing** —
  not real AWQModifier fields.
- Every VLM config (`modality: vision_text`) couldn't instantiate.
- `config_smooth` (`method: smoothquant`) would have ValidationError'd.
- `config_vlm_nanollava`: `method: awq` + `scheme: W8A8` — invalid (AWQ is
  W4A16-only); fixed to `gptq`.

## 5. Config matrix rebuild — `config/quantize/`

Cleared all old quant YAMLs. Rebuilt per-model subdirs.

### `qwen3_4b_thinking/` — Qwen/Qwen3-4B-Thinking-2507 (LLM), calib `ultrachat_200k`
10 configs:
- `llmc_awq_w4a16`, `llmc_gptq_w4a16`, `llmc_gptq_w8a8`,
  `llmc_ptq_w8a8_smoothquant`, `llmc_ptq_fp8`, `llmc_ptq_fp8_dynamic`,
  `llmc_ptq_nvfp4`
- `modelopt_ptq_fp8`, `modelopt_awq_nvfp4` (gs 16), `modelopt_ptq_mxfp4` (gs 32)

### `qwen2_5_vl_3b/` — Qwen/Qwen2.5-VL-3B-Instruct (VLM), calib `linxy/LaTeX_OCR`
5 configs: `llmc_{awq_w4a16, gptq_w4a16, gptq_w8a8, ptq_fp8, ptq_fp8_dynamic}`

### `hunyuan_ocr/` — tencent/HunyuanOCR (1B OCR VLM), calib `linxy/LaTeX_OCR`
5 configs: same 5 schemes. `trust_remote_code: true`.

VLMs are llm_compressor-only: modelopt's calibration loop assumes a chat
`messages` field, incompatible with OCR image-text datasets. Vision tower
auto-excluded by `BaseQuantizer._merged_ignore`.

## 6. Benchmark configs — `config/benchmark/`

3 new, one per model — FP16 baseline + every quantized variant, paths set to the
exact `{model}-{backend}-{method}-{scheme}` save subfolders. `runtimes: ["hf"]`.
- `qwen3_4b_thinking.yaml` — quality_llm [ppl, mmlu_tiny] + perf + memory.
- `qwen2_5_vl_3b.yaml` — quality_ocr [cer, wer, exact_match] + perf + memory.
- `hunyuan_ocr.yaml` — same OCR metrics; `trust_remote_code: true` on all entries.
- NVFP4/MXFP4 entries marked `skip_on: ["sm_86"]` (Blackwell-only).

## 7. Validation done

- All 20 quant configs: `model_validate` + `_build_recipe()` pass.
- Negative tests: gptq+FP8, awq+W8A8, FP8+awq, MXFP4-on-llmc, bad AWQ/scheme
  fields all fail loudly (as intended).
- All 3 benchmark configs: `benchmark.py --dry-run` → "config valid ✓".

## 8. Caveats / open

- **HunyuanOCR**: custom arch `HunYuanVLForConditionalGeneration`; model card
  pins a specific transformers commit + wants processor `use_fast=False`.
  `base.py` AutoProcessor doesn't pass `use_fast` → load may fail; runtime issue,
  not config.
- FP8/NVFP4/MXFP4 inference needs Hopper/Blackwell — quantization/export works on
  Ampere, benchmark load will fail on 3060 (crash-safe per-model handles it).
- Old `config/benchmark/{llm,ocr}_comparison.yaml` now orphaned (point to deleted
  TinyLlama outputs) — decide delete vs repoint.
- Not committed yet.

## Files touched
- `src/quantization/llm_compressor.py` (recipe rewrite + cleanup)
- `src/config/schemas.py` (enums, validators, extra=forbid)
- `config/quantize/**` (cleared + 20 new across 3 subdirs)
- `config/benchmark/{qwen3_4b_thinking,qwen2_5_vl_3b,hunyuan_ocr}.yaml` (new)
