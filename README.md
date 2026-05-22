# TripleQuant-VLM

Production-grade quantization + benchmarking pipeline for LLMs and Vision-Language Models. Supports AWQ, GPTQ, PTQ, and SmoothQuant via dual backends (`llmcompressor` and `modelopt`), with a unified YAML-driven CLI and modality-aware calibration.

---

## Architecture

```
TripleQuant-VLM/
├── quantize.py                  # CLI: quantize a model from YAML config
├── tests/
│   └── simple_generate.py       # Load + test / compare / chat with quantized model
├── src/
│   ├── config/
│   │   ├── schemas.py           # Pydantic v2: QuantizeConfig, SchemeConfig, CalibrationConfig …
│   │   └── loader.py            # YAML → validated config
│   ├── quantization/
│   │   ├── base.py              # BaseQuantizer: VLM/LLM loader, save()
│   │   ├── factory.py           # get_quantizer() — routes on config.backend
│   │   ├── llm_compressor.py    # AWQ / GPTQ / PTQ / SmoothQuant via llmcompressor
│   │   ├── modelOpy.py          # FP8 / NVFP4 / MXFP4 / INT4/INT8 via nvidia-modelopt
│   │   └── registry.py          # @register("method") decorator registry
│   ├── data/                    # calibration + dataloader (in progress)
│   └── integrations/            # W&B logger (in progress)
├── config/quantize/             # Ready-to-run YAML configs (see below)
└── notes/
    ├── plan.md                  # 3-week sprint plan + current-state audit
    ├── benchmark.md             # Benchmark pipeline design (W&B / Langfuse / MLflow)
    └── turboquant.md            # TurboQuant algorithm + Triton kernel plan
```

---

## Backends

| Backend | Methods | Schemes | vLLM load flag |
|---|---|---|---|
| `llm_compressor` | AWQ, GPTQ, PTQ, SmoothQuant | W4A16, W4A16_ASYM, W8A8, W8A16, FP8, FP8_DYNAMIC | `compressed-tensors` |
| `modelopt` | AWQ, PTQ | W4A16, W8A8, W8A16, FP8, FP8_DYNAMIC, FP8_BLOCK, NVFP4, MXFP4 | `modelopt` / `modelopt_fp4` |

**Hardware floor:**

| Scheme | Min GPU |
|---|---|
| INT4 / INT8 | Any CUDA ≥ Ampere |
| FP8 | Hopper (H100/H200), Ada (L40S) — emulated on Ampere |
| NVFP4 | Blackwell (B100, RTX 50xx) |
| MXFP4 | Blackwell native; Hopper emulated |

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install llmcompressor compressed-tensors pydantic>=2 transformers datasets pyyaml
# For modelopt backend (optional):
pip install nvidia-modelopt[torch]
```

---

## 1. Quantize a Model

```bash
python quantize.py --config config/quantize/config_1b_llmcompressor.yaml
```

Output lands in `output_dir` (from YAML) under a descriptive subfolder:
```
{output_dir}/{model_name}-{backend}-{method}-{scheme}/
```

### Available configs

| Config | Model | Backend | Method | Scheme |
|---|---|---|---|---|
| `config_1b_llmcompressor.yaml` | TinyLlama-1.1B | llm_compressor | GPTQ | W8A8 + SmoothQuant |
| `config_1b_modelopt.yaml` | TinyLlama-1.1B | modelopt | AWQ | W4A16 |
| `config_vlm_nanollava.yaml` | nanoLLaVA-1.5 | llm_compressor | AWQ | W8A8 |
| `qwen_2_5vl_3B_ocr_awq_llm_compressor.yaml` | Qwen2.5-VL-3B | llm_compressor | AWQ | W4A16 |
| `qwen_2_5vl_3B_ocr_gptq_llm_compressor.yaml` | Qwen2.5-VL-3B | llm_compressor | GPTQ | W4A16 |
| `qwen_2_5vl_ocr_modelOPT.yaml` | Qwen2.5-VL-7B | modelopt | AWQ | W4A16 |
| `SmolVLM2-2.2B-Instruct_ocr_llm_compressor.yaml` | SmolVLM2-2.2B | llm_compressor | GPTQ | W4A16 |

### Config format

```yaml
method: gptq           # awq | gptq | turboquant (planned)
backend: llm_compressor

model:
  model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  torch_dtype: bfloat16
  device_map: auto
  model_type: llm      # llm | vlm | moe

scheme:
  scheme: W8A8
  group_size: 128
  symmetric: true
  observer: mse        # mse | minmax | maxabs | percentile
  per_channel: false
  targets: ["Linear"]
  ignore: ["lm_head"]

calibration:
  dataset_name: HuggingFaceH4/ultrachat_200k
  split: train_sft
  num_samples: 512
  max_seq_len: 2048
  dataset_format: auto  # auto | chat | image_text

output:
  output_dir: ./my-output
  save_compressed: true
  save_processor: true

smoothquant:           # optional — only valid with A8 activation schemes
  enabled: true
  strength: 0.5

gptq:
  dampening_frac: 0.01
```

**VLM note:** Vision tower modules are automatically excluded from quantization. No manual ignore patterns needed for `visual.*`, `vision_tower.*`, `merger.*`, etc.

---

## 2. Test a Quantized Model

```bash
# Auto-detects compressed format from config.json
python tests/simple_generate.py --model ./output/TinyLlama-1.1B-Chat-v1.0-llm_compressor-gptq-W8A8

# Compare quantized vs FP16 baseline
python tests/simple_generate.py \
  --model ./output/TinyLlama-1.1B-Chat-v1.0-llm_compressor-gptq-W8A8 \
  --baseline TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Interactive chat
python tests/simple_generate.py \
  --model ./output/TinyLlama-1.1B-Chat-v1.0-llm_compressor-gptq-W8A8 \
  --interactive

# Load uncompressed / HF model
python tests/simple_generate.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --no-compressed
```

Reports: load time, disk size, VRAM usage, tok/s per prompt.

---

## 3. Serve with vLLM

```bash
# llmcompressor output (compressed-tensors format)
vllm serve ./output/TinyLlama-W8A8 --quantization compressed-tensors

# modelopt FP8 output
vllm serve ./output/Qwen2.5-VL-7B-FP8 --quantization modelopt

# modelopt NVFP4 output (Blackwell only)
vllm serve ./output/model-nvfp4 --quantization modelopt_fp4
```

---

## 4. Extending

**Add a new quantization method:**

```python
# src/quantization/my_method.py
from src.quantization.registry import register
from src.quantization.base import BaseQuantizer

@register("my_method")
class MyQuantizer(BaseQuantizer):
    def quantize(self) -> None:
        ...
```

Add `"my_method"` to `MethodLiteral` in `src/config/schemas.py`. Done — factory picks it up automatically.

**Add a new model architecture:**

Ignore patterns for vision towers and unsupported SSM layers are auto-detected in `llm_compressor.py::_get_vision_ignore_patterns`. For exotic architectures, override `ignore` in the YAML `scheme.ignore` list.

---

## Roadmap

See `notes/plan.md` for the full day-by-day 3-week sprint. High-level:

### Week 1 (pipeline polish) — in progress
- [ ] `fp16.py` baseline quantizer (`@register("fp16")`)
- [ ] Fix `BenchmarkConfig` schema + `loader.py` import error
- [ ] Enable `modelopt` backend in factory (currently commented out)
- [ ] Fix NVFP4/MXFP4 config dicts to use `mtq.NVFP4_DEFAULT_CFG` (not hand-rolled)
- [ ] Arch profiles (`arch_profiles.py`) — MoE sequential targets, Mamba ignore
- [ ] Benchmark harness (`benchmark.py`) — PPL, CER, TTFT, TPOT, VRAM, disk size
- [ ] Auto-enqueue: successful quantize run → appended to `benchmark_queue.yaml`
- [ ] Eval tracking: W&B (plots + tables), Langfuse (OCR per-sample traces), MLflow (registry)
- [ ] VLM eval: CER/WER on LaTeX_OCR for Qwen2.5-VL configs

### Week 2 (TurboQuant — KV-cache quantization)
- [ ] `src/turboquant/` — Lloyd-Max codebook, orthogonal rotation, bit-packing
- [ ] `TurboQuantMSE` + `TurboQuantProd` (MSE + 1-bit QJL residual)
- [ ] KV-cache capture hooks for HF Llama-family models
- [ ] Python reference decode-step attention over compressed KV
- [ ] Correctness gate: wikitext PPL delta ≤ 0.3 at K=3-bit / V=2-bit

### Week 3 (Triton kernels)
- [ ] Kernel A: fused MSE score `⟨q_rot, centroids[idx]⟩` — no K materialization
- [ ] Kernel B: fused QJL score `⟨S q, signs⟩`
- [ ] Kernel C: fused decode — online softmax + value gather, one HBM pass over KV
- [ ] HF `generate` shim for end-to-end generation with TurboQuant KV
- [ ] Benchmark: FP16 vs FP8-KV vs TurboQuant at ctx {1K, 8K, 32K}

### Deferred (post-sprint)
- Minitron pruning (depth + width) via `nvidia-modelopt`
- Knowledge distillation post-prune (80–100B token recipe)
- AutoQuantize per-layer precision search
- SVDQuant (low-rank + 4-bit)
- 2:4 structured sparsity
- TensorRT-LLM export
- Audio modality calibration

---

## License

Apache 2.0
