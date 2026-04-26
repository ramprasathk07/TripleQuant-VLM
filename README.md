# TripleQuant-VLM

Production-grade AWQ/GPTQ weight quantization + benchmarking for Vision-Language Models (e.g. Qwen2.5-VL), featuring `vLLM` serving, CER/WER OCR evaluation, and structured experiment tracking.

## Architecture

TripleQuant-VLM uses a modular, Pydantic-validated architecture to keep quantization and benchmarking workflows strict and reproducible.

```text
TripleQuant-VLM/
├── configs/
│   ├── quantize/           # One file = one model + one quant method
│   └── benchmark/          # One file = N models to compare sequentially
├── src/
│   ├── config/             # Pydantic schemas and YAML loaders
│   ├── data/               # VLM dataset processors
│   ├── evaluation/         # Latency (vLLM), Accuracy (WER), Memory
│   └── quantization/       # Registry-based AWQ/GPTQ via llmcompressor
├── quantize.py             # CLI for quantization
└── benchmark.py            # CLI for benchmarking
```

## Installation

Install the package in editable mode with all dependencies:

```bash
pip install -e .
```

*Note: You may need to install `torch` and `torchvision` matching your CUDA version beforehand if you are on Windows or a custom cluster.*

## 1. Quantization Workflow

The `quantize.py` script applies a specific quantization recipe to a single model using a calibration dataset.

**Configuration:**
Look at `configs/quantize/qwen25vl_3b_awq.yaml` for an example. It defines the model, quantization method (e.g. `awq` or `gptq`), calibration parameters, and output paths.

**Usage:**
```bash
# Validate the config without allocating GPU memory
python quantize.py --config configs/quantize/qwen25vl_3b_awq.yaml --dry-run

# Run the quantization (saves weights to `outputs/`)
python quantize.py --config configs/quantize/qwen25vl_3b_awq.yaml
```

*Adding new methods:* The quantizers use a decorator registry. To add a new method, inherit from `BaseQuantizer`, decorate it with `@register("my_method")` inside `src/quantization/`, and the config parser will automatically pick it up.

## 2. Benchmark Workflow

The `benchmark.py` script evaluates multiple models (e.g., an FP16 baseline against your quantized variants) sequentially in a crash-safe manner.

**Configuration:**
Look at `configs/benchmark/ocr_comparison.yaml`. It takes a list of `models`, a dataset to evaluate against, and a matrix of `metrics` (Latency, Accuracy, Memory, Perplexity).

**Usage:**
```bash
python benchmark.py --config configs/benchmark/ocr_comparison.yaml
```

**Features:**
- **Crash-safe:** Results are saved to `results/` immediately after each model finishes. If one model OOMs, the pipeline logs the error and moves to the next.
- **vLLM Integration:** Uses `vLLM` to calculate true serving metrics like TTFT (Time To First Token) and TPOT (Time Per Output Token).
- **Automated Summary:** Produces a final `comparison_summary.json` with a side-by-side terminal table.

## License

Apache 2.0
