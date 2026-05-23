# Benchmark Pipeline — Design & Implementation Plan

**Updated:** 2026-05-22
**Scope:** unified eval harness for quantized models from both `llmcompressor` and `modelopt` backends. Two hardware targets: **RTX 3060 (12GB Ampere)** and **AMD MI300X (192GB CDNA3, ROCm)**. Two task families: **LLM text** (PPL + MMLU-tiny) and **OCR** (CER/WER on LaTeX_OCR, TextOCR).

Companion docs: `notes/plan.md` (sprint schedule), `notes/turboquant.md` (kernel work).

---

## 1. Goals

1. One CLI: `python benchmark.py --config <yaml>` — runs N models sequentially, crash-safe.
2. **Auto-enqueue:** every successful `quantize.py` run appends model path to a persistent queue (`benchmark_queue.yaml`). Quantize 10 models → all benchmarked in one shot.
3. Cross-backend, cross-hardware parity: same metrics computed identically whether model came from llmcompressor or modelopt, ran on 3060 or MI300X.
4. Outputs structured JSON + summary CSV + auto-generated plots (PNG via matplotlib local), **plus pushed to public/shared trackers: W&B (primary), Langfuse (LLM-trace), MLflow (artifact + metric registry)**.
5. Reproducible: fixed seeds, version-pinned deps, hardware fingerprint in every result file + tracker run config.
6. Every plot, metric, and per-sample OCR prediction available via public W&B project URL — shareable link, no local deps to view.

---

## 2. Hardware Constraints

### RTX 3060 (12GB, Ampere sm_86)

| Capability | Status |
|---|---|
| INT4 W4A16 (Marlin) | ✅ native, fast |
| INT8 W8A8 | ✅ native |
| W4A8 INT | ⚠️ supported but slower than W4A16 |
| FP8 E4M3/E5M2 | ❌ no native — vLLM emulates (Triton fallback, slow + sometimes broken on sm_86) |
| NVFP4 / MXFP4 | ❌ no Blackwell hardware |
| Max model | ~7B FP16, ~13B INT4 (with ctx ≤4K). 8B INT4 fits comfortably. |
| KV-cache headroom | very tight — TurboQuant most impactful here |
| Runtime | vLLM CUDA, transformers, exllama2 (Marlin) |

**3060 test matrix:** TinyLlama-1.1B, Qwen2.5-1.5B/3B, Llama-3-8B (INT4 only), SmolVLM2-2.2B, Qwen2.5-VL-3B.

### AMD MI300X (192GB HBM3, CDNA3, gfx942)

| Capability | Status |
|---|---|
| FP8 W8A8 (E4M3) | ✅ native AITER kernels |
| INT4 / INT8 GPTQ + AWQ via Marlin | ✅ via vLLM ROCm |
| MXFP4 weight-only | ✅ vLLM ≥ 0.14 (no act quant yet) |
| NVFP4 | ❌ NVIDIA-only |
| compressed-tensors loader | ✅ supported, recent fix for DeepSeek-R1 |
| modelopt FP8 export | ✅ via vLLM `quantization="modelopt"` |
| Max model | 70B FP16 fits, 405B INT4 fits |
| Runtime | vLLM ROCm Docker image (prebuilt as of Jan 2026) |

**MI300X test matrix:** same as 3060 + Llama-3-70B, Qwen2.5-VL-32B/72B, MoE (Mixtral-8x7B).

**Cross-hardware comparison rule:** only configs that load on *both* feed the parity table. Big models marked MI300X-only.

---

## 3. Metric Catalog

### 3.1 Quality — LLM text

| Metric | Dataset | What it measures | Cost |
|---|---|---|---|
| **Perplexity (PPL)** | wikitext-2-raw-v1 test, ctx 2048, stride 512 | Language modeling fidelity. Standard. | ~5 min for 8B model on 3060 |
| **MMLU-tiny** | MMLU val subset (500 q across 5 subjects) | Reasoning + factual recall delta | ~10 min |
| **Logit-KL vs FP16** | 256 wikitext prompts | Token-level distribution shift | ~2 min |
| **Token-agreement %** | greedy 64 tokens on 50 prompts | Coarse "did it break" | ~2 min |

PPL is the gate. MMLU-tiny + logit-KL are diagnostics — surface where PPL hides damage.

### 3.2 Quality — OCR (VLM)

| Metric | Dataset | What it measures |
|---|---|---|
| **CER** (char error rate) | linxy/LaTeX_OCR test, 500 samples | Char-level edit distance / ref len. Primary. |
| **WER** (word error rate) | LaTeX_OCR + TextOCR sample | Token/word level |
| **Exact-match %** | LaTeX_OCR | Strict; useful for formula correctness |
| **BLEU-4** | TextOCR captioned | Phrase-level fluency |

CER is primary. Compute via `jiwer` (text) and a tex-tokenizer normalizer for LaTeX (strip whitespace, collapse `\,`).

### 3.3 Performance — Latency & throughput (vLLM serving)

| Metric | Measurement |
|---|---|
| **TTFT** (time-to-first-token) | median + p95 over 100 requests, prompt 512 tok |
| **TPOT** (time-per-output-token) | median + p95, decode 128 tok |
| **End-to-end latency** | TTFT + decode_time |
| **Throughput (tok/s)** | total decode tokens / wall time, single request |
| **Concurrent throughput** | batch sizes {1, 4, 8, 16, 32} until OOM |
| **Prefill throughput** | tokens/s of prompt processing |

Measured via `vllm.LLM` offline batched API + `time.perf_counter` (avoids server overhead noise). Background HTTP load uses `vllm.entrypoints.openai.api_server` + `wrk` only if user passes `--serve-mode`.

### 3.4 Performance — Context-length sweep

Per model, run ctx ∈ {512, 2048, 8192, 32768} (truncate if model maxlen < value). Report tok/s + peak VRAM. Catches KV-cache scaling cliffs — critical for TurboQuant comparison later.

### 3.5 Memory & size

| Metric | Source |
|---|---|
| **Model size on disk (MB)** | sum `.safetensors` + `.bin` + `.pt` |
| **Peak VRAM allocated** | `torch.cuda.max_memory_allocated()` after generation |
| **Peak VRAM reserved** | `torch.cuda.max_memory_reserved()` |
| **KV-cache footprint** | `n_layers × n_kv_heads × head_dim × ctx × 2 × dtype_bytes` (computed, not measured) |
| **Load time (s)** | wall time of `from_pretrained` |

MI300X uses `torch.cuda.*` under ROCm (PyTorch HIP shim — same API).

### 3.6 Derived / Pareto

| Metric | Formula |
|---|---|
| **Quality preservation** | `1 - (PPL_quant - PPL_fp16) / PPL_fp16` |
| **Compression ratio** | `size_fp16 / size_quant` |
| **Speedup** | `tput_quant / tput_fp16` |
| **Efficiency score** | `quality_pres × speedup × compression` — one-number ranking |

---

## 4. Pipeline Architecture

```
quantize.py  ───── on success ─────►  benchmark_queue.yaml  (append)
                                              │
                                              ▼
                                     benchmark.py --queue
                                              │
                  ┌───────────────────────────┼───────────────────────────┐
                  ▼                           ▼                           ▼
            Eval: LLM text             Eval: OCR VLM             Eval: latency
            (PPL, MMLU, KL)            (CER, WER, EM)            (TTFT, TPOT, tput)
                  │                           │                           │
                  └───────────────────────────┼───────────────────────────┘
                                              ▼
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
      LOCAL                          WANDB (primary)                  LANGFUSE
      results/{run_id}/*.json        wandb.log per metric             trace per OCR sample
      results/{run_id}/plots/*.png   wandb.Image for every plot       prompt+pred+CER score
      summary.csv                    wandb.Table for summary CSV      project=triplequant-ocr
                                     wandb.Artifact for model         shareable trace URL
                                              │
                                              ▼
                                          MLFLOW
                                  mlflow.log_metric (cross-tool)
                                  mlflow.log_artifact (CSV + PNG)
                                  experiment=triplequant-bench
                                  registers quantized model paths
                                              │
                                              ▼
                          benchmark_aggregate.py → cross-run W&B Report (auto-generated)
                                                 + MLflow comparison view URL
```

### 4.1 File layout

```
src/evaluation/
  __init__.py
  runner.py             # orchestrator — picks evaluators per model_type
  eval_llm.py           # PPL, MMLU-tiny, logit-KL, token-agreement
  eval_ocr.py           # CER, WER, exact-match, BLEU
  eval_latency.py       # TTFT, TPOT, tput via vllm.LLM
  eval_memory.py        # disk size, VRAM, load time
  hardware.py           # detect GPU (sm_xx / gfx942), CUDA/ROCm, RAM
  plots.py              # matplotlib bar/line/Pareto plots → PNG + bytes-buffer for upload
  aggregate.py          # walk results/ → CSV + cross-model comparison
  utils.py              # text normalization for OCR, sample loaders
src/tracking/           # NEW — unified tracker abstraction
  __init__.py
  base.py               # TrackerBase ABC: log_metric, log_plot, log_artifact, log_table, log_trace
  wandb_tracker.py      # W&B impl (primary)
  langfuse_tracker.py   # Langfuse impl (per-sample LLM trace)
  mlflow_tracker.py     # MLflow impl (cross-run metric/artifact registry)
  composite.py          # CompositeTracker — fanout to multiple backends
  noop.py               # NoOpTracker for offline / CI
benchmark.py            # CLI entry
src/config/schemas.py   # + BenchmarkConfig, TrackingConfig, BenchmarkModelEntry, MetricsConfig
config/benchmark/
  ocr_comparison.yaml
  llm_perf.yaml
  full_sweep.yaml
benchmark_queue.yaml    # auto-populated by quantize.py; user can edit/clear
results/
  {run_id}/
    {model_id}.json            # per-model raw metrics
    summary.csv                # all models flat table
    tracker_urls.json          # NEW — {wandb_run_url, mlflow_run_id, langfuse_session_url}
    plots/
      ppl_bar.png
      cer_bar.png
      latency_vs_size.png
      throughput_vs_batch.png
      pareto_quality_speed.png
      ctx_sweep_throughput.png
    hardware.json              # GPU + driver + cuda/rocm version
    run_log.txt
```

### 4.2 BenchmarkConfig schema (Pydantic)

```python
class MetricsConfig(BaseModel):
    quality_llm: list[Literal["ppl", "mmlu_tiny", "logit_kl", "token_agree"]] = ["ppl"]
    quality_ocr: list[Literal["cer", "wer", "exact_match", "bleu"]] = ["cer"]
    perf:        list[Literal["ttft", "tpot", "throughput", "ctx_sweep"]] = ["throughput", "ttft", "tpot"]
    memory:      list[Literal["disk", "vram", "load_time"]] = ["disk", "vram"]

class EvalDatasetConfig(BaseModel):
    ppl_dataset: str = "wikitext"
    ppl_subset: str = "wikitext-2-raw-v1"
    mmlu_subjects: list[str] = ["high_school_mathematics", "computer_science",
                                "philosophy", "world_history", "global_facts"]
    ocr_dataset: str = "linxy/LaTeX_OCR"
    ocr_num_samples: int = 500
    ocr_max_new_tokens: int = 256

class LatencyConfig(BaseModel):
    prompt_lens: list[int] = [512]
    output_lens: list[int] = [128]
    batch_sizes: list[int] = [1, 4, 8, 16]
    ctx_sweep:   list[int] = [512, 2048, 8192]
    num_requests: int = 100
    warmup_requests: int = 5

class BenchmarkModelEntry(BaseModel):
    name: str
    path: str                     # local dir or HF id
    is_local: bool = False
    is_compressed: bool = True    # save_compressed=True flag
    model_type: ModelTypeLiteral = "llm"   # 'llm' | 'vlm'
    backend_hint: Optional[Literal["llm_compressor", "modelopt"]] = None
    vllm_quantization: Optional[str] = None  # 'compressed-tensors', 'modelopt', 'modelopt_fp4'
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    skip_on: list[str] = []       # e.g., ["sm_86"] to skip on 3060

class TrackingConfig(BaseModel):
    """Multi-backend experiment tracking. Local PNG always written; trackers are additive."""
    enabled: list[Literal["wandb", "langfuse", "mlflow"]] = ["wandb", "mlflow"]

    # W&B
    wandb_project: str = "triplequant-vlm"
    wandb_entity: Optional[str] = None
    wandb_tags: list[str] = []
    wandb_public: bool = True              # use public W&B project (shareable URL)
    wandb_api_key_env: str = "WANDB_API_KEY"

    # Langfuse — for OCR per-sample LLM traces (prompt, image, pred, CER score)
    langfuse_project: str = "triplequant-ocr"
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    langfuse_secret_key_env: str = "LANGFUSE_SECRET_KEY"
    langfuse_only_ocr: bool = True         # don't trace PPL eval (too many calls, no value)

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"     # local default; can be http://mlflow-server
    mlflow_experiment: str = "triplequant-bench"
    mlflow_register_model: bool = False    # only set True for "release" runs

    # Common
    log_per_sample_predictions: bool = False  # writes every OCR pred to tracker (heavy)
    log_artifacts: bool = True              # PNG + CSV uploaded
    offline_mode: bool = False              # fall back to NoOpTracker if creds missing

class BenchmarkConfig(BaseModel):
    run_name: str
    output_root: Path = Path("./results")
    models: list[BenchmarkModelEntry]
    baseline: Optional[BenchmarkModelEntry] = None  # for delta metrics
    metrics: MetricsConfig = MetricsConfig()
    datasets: EvalDatasetConfig = EvalDatasetConfig()
    latency: LatencyConfig = LatencyConfig()
    tracking: TrackingConfig = TrackingConfig()
    crash_safe: bool = True
    seed: int = 42
    hf_token_env: str = "HF_TOKEN"
```

### 4.3 Per-model output JSON

```jsonc
{
  "model": {"name": "...", "path": "...", "size_mb": 4123.5},
  "hardware": {"gpu": "AMD MI300X", "arch": "gfx942", "vram_gb": 192,
               "runtime": "rocm-6.2", "vllm": "0.14.0"},
  "load": {"time_s": 12.4, "vram_after_mb": 4521.0},
  "quality": {
    "llm":  {"ppl": 6.82, "mmlu_tiny_acc": 0.61,
             "logit_kl_mean": 0.018, "token_agree_pct": 96.5},
    "ocr":  {"cer": 0.082, "wer": 0.21, "exact_match": 0.45, "bleu4": 0.71}
  },
  "perf": {
    "ttft_ms_p50": 88.1, "ttft_ms_p95": 102.4,
    "tpot_ms_p50": 6.2,  "tpot_ms_p95": 7.5,
    "throughput_tok_s": [{"batch": 1, "tput": 145.2}, {"batch": 8, "tput": 980.1}],
    "ctx_sweep": [{"ctx": 512, "tput": 152.0, "vram_mb": 4800},
                  {"ctx": 8192, "tput": 88.0, "vram_mb": 7600}]
  },
  "memory": {"peak_vram_mb": 7600, "peak_reserved_mb": 8100, "disk_mb": 4123.5},
  "delta_vs_baseline": {"ppl_delta": 0.14, "speedup": 2.31,
                        "compression": 4.0, "quality_pres": 0.978},
  "errors": [],
  "timestamp": "2026-05-29T14:21:33Z"
}
```

---

## 5. Auto-Enqueue: Quantize → Benchmark Wire

### 5.1 Trigger point

End of `LLMCompressorQuantizer.quantize()` and `ModelOptQuantizer.quantize()`, **only on success**, after `self.save(...)`:

```python
from src.evaluation.queue import enqueue_for_benchmark
enqueue_for_benchmark(
    output_dir=self.save_path,       # actual subfolder, not parent
    config=self.config,
    backend=self.config.backend,
)
```

Wrap in try/except — queue failure must not fail the quant run. Log warning, continue.

### 5.2 Queue file format (`benchmark_queue.yaml`)

```yaml
queue:
  - name: TinyLlama-1.1B-llm_compressor-gptq-W8A8
    path: ./tinyllama-1b-awq-llmcompressor/TinyLlama-1.1B-Chat-v1.0-llm_compressor-gptq-W8A8
    is_local: true
    is_compressed: true
    model_type: llm
    backend_hint: llm_compressor
    vllm_quantization: compressed-tensors
    added_at: 2026-05-22T17:45:01Z
    source_config: config/quantize/config_1b_llmcompressor.yaml
    status: pending           # pending | running | done | failed
  - name: Qwen2.5-VL-3B-llm_compressor-awq-W4A16
    path: ./qwen2.5vl-awq-w4a16-llmcompressor/...
    model_type: vlm
    ...
```

`status` updated by benchmark runner. On `done`, optionally archive to `benchmark_queue.archive.yaml` to keep main queue short.

### 5.3 CLI flows

```bash
# Standard: read queue, run all pending, append to ./results/<run_id>/
python benchmark.py --queue

# One-shot: explicit config (queue ignored)
python benchmark.py --config config/benchmark/full_sweep.yaml

# Dry-run: validate queue + show plan, no eval
python benchmark.py --queue --dry-run

# Clear queue manually
python benchmark.py --clear-queue

# Re-run failed only
python benchmark.py --queue --retry-failed
```

### 5.4 vllm_quantization auto-detect

Reading the saved checkpoint's `config.json`:
- `quantization_config.format == "pack-quantized"` → `vllm_quantization = "compressed-tensors"`
- `hf_quant_config.json` exists → check `quant_algo`:
  - `FP8` → `"modelopt"`
  - `NVFP4` → `"modelopt_fp4"`
  - `MXFP8` → `"modelopt_mxfp8"`
- else → None (use default vLLM detection)

Helper `src/evaluation/detect.py::detect_vllm_quant_arg(model_path) -> str | None`. Called both at enqueue time (snapshot) and at load time (re-verify).

---

## 6. Evaluator Implementations (sketch)

### 6.1 PPL — sliding window

```python
def compute_ppl(model, tokenizer, dataset, ctx=2048, stride=512):
    # Standard HF sliding-window PPL: see transformers PPL example
    # Encode whole corpus → for each window: shift by stride, compute NLL on new tokens
    nlls = []
    enc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    for begin in range(0, enc.input_ids.size(1), stride):
        end = min(begin + ctx, enc.input_ids.size(1))
        trg_len = end - max(begin + ctx - stride, 0)
        ids = enc.input_ids[:, begin:end].to(model.device)
        targets = ids.clone(); targets[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(ids, labels=targets).loss
        nlls.append(loss * trg_len)
    return torch.exp(torch.stack(nlls).sum() / end).item()
```

### 6.2 MMLU-tiny

Load `cais/mmlu`, filter to 5 subjects × 100 q. Format prompt as multiple-choice with answer letter, score by logit of `A/B/C/D` token at generation step 0. Closed-form, no generation needed.

### 6.3 OCR CER

```python
import jiwer
preds = vlm_model.generate(image, prompt="Convert to LaTeX formula.")
cer = jiwer.cer(refs_normalized, preds_normalized)
```

LaTeX normalizer: strip whitespace, collapse `\,` `\;` `\ `, lowercase command names. WER same with `jiwer.wer`.

### 6.4 Latency — vLLM offline

```python
from vllm import LLM, SamplingParams
llm = LLM(model=path, quantization=detected, gpu_memory_utilization=0.85,
          max_model_len=cfg.max_model_len, dtype="auto")
# TTFT: SamplingParams(max_tokens=1)
# TPOT: SamplingParams(max_tokens=128, ignore_eos=True)
# wall = perf_counter; ttft from first-token callback if vLLM exposes; else single-token-gen
```

For batch sweep: build `[prompt] * batch_size`, call `llm.generate(...)` once, divide. Repeat 5 times, take median.

### 6.5 Dual Runtime — vLLM + HF Transformers (parallel eval)

Every model evaluated **twice**: once via `vllm.LLM` (production path) and once via raw `transformers.AutoModelForCausalLM` / `AutoModelForVision2Seq` (reference path). Quality should match within tolerance; perf will differ massively. Captures kernel/runtime overhead separately from quant scheme cost.

#### 6.5.1 Why both

| Concern | vLLM | HF transformers |
|---|---|---|
| Production realism | ✅ what users hit | ❌ slow eager path |
| Sanity check (quant arithmetic) | ⚠️ Marlin/AITER kernels can mask bad quant | ✅ pure PyTorch math |
| Coverage of quant formats | ✅ compressed-tensors, modelopt, AWQ, GPTQ, FP8, NVFP4, MXFP4 | ⚠️ only what `transformers` natively loads (bitsandbytes, GPTQ via auto-gptq, AWQ via autoawq, compressed-tensors via `compressed-tensors` HF integration) |
| Per-sample logit access | ❌ vLLM hides logits | ✅ trivial — needed for logit-KL, MMLU |
| Long ctx perf | ✅ PagedAttention | ❌ KV grows linearly, OOM fast |
| Batch ≥ 4 | ✅ continuous batching | ⚠️ static batch only |
| TTFT/TPOT measurement | ✅ first-class | ⚠️ need manual streamer hook |

Rule: **HF runtime owns quality metrics** (PPL, MMLU, logit-KL, token-agree, CER). **vLLM owns perf metrics** (TTFT, TPOT, throughput, batch sweep, ctx sweep). Cross-validate quality via a small overlap (top-50 CER samples run on both).

#### 6.5.2 Config schema additions

```python
class RuntimeConfig(BaseModel):
    """Which runtimes to evaluate against. Both default-on for parity."""
    vllm: bool = True
    hf: bool = True
    # quality always uses hf when enabled; vllm when hf disabled
    perf_runtime: Literal["vllm", "hf", "both"] = "vllm"
    quality_runtime: Literal["hf", "vllm", "both"] = "hf"

    # HF loader knobs
    hf_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    hf_device_map: str = "auto"
    hf_attn_impl: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    hf_trust_remote_code: bool = True

    # vLLM loader knobs (per-model overrides via BenchmarkModelEntry)
    vllm_dtype: Literal["auto", "float16", "bfloat16"] = "auto"
    vllm_enforce_eager: bool = False          # disable cudagraph for clean prof
    vllm_disable_log_stats: bool = True
    vllm_swap_space_gb: int = 4
    vllm_kv_cache_dtype: Literal["auto", "fp8", "fp8_e5m2"] = "auto"

class BenchmarkConfig(BaseModel):
    ...
    runtime: RuntimeConfig = RuntimeConfig()
```

Add `runtime` field to `BenchmarkConfig`. Existing entry already carries `vllm_quantization` — pass through.

#### 6.5.3 Loader abstraction

```
src/evaluation/runtimes/
  __init__.py
  base.py             # RuntimeBase ABC
  vllm_runtime.py     # VLLMRuntime
  hf_runtime.py       # HFRuntime
  factory.py          # build_runtime(name, entry, cfg) -> RuntimeBase
```

```python
class RuntimeBase(ABC):
    name: str
    @abstractmethod
    def load(self, entry: BenchmarkModelEntry) -> None: ...
    @abstractmethod
    def generate(self, prompts: list[str], max_new_tokens: int,
                 temperature: float = 0.0) -> list[str]: ...
    @abstractmethod
    def generate_with_logits(self, prompts: list[str], max_new_tokens: int
                             ) -> tuple[list[str], list[Tensor]]: ...   # HF only — vLLM raises NotImplementedError
    @abstractmethod
    def forward_logits(self, input_ids: Tensor) -> Tensor: ...           # for PPL / KL
    @abstractmethod
    def generate_vlm(self, image, prompt: str, max_new_tokens: int) -> str: ...
    @abstractmethod
    def measure_ttft_tpot(self, prompt: str, n: int = 100) -> dict: ...
    @abstractmethod
    def measure_throughput(self, prompt: str, batch_sizes: list[int],
                           output_len: int) -> list[dict]: ...
    @abstractmethod
    def peak_vram_mb(self) -> float: ...
    @abstractmethod
    def unload(self) -> None: ...                                        # explicit destroy_model_parallel + gc + empty_cache
```

#### 6.5.4 `HFRuntime` — quality eval owner

```python
class HFRuntime(RuntimeBase):
    name = "hf"
    def load(self, entry):
        from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
        cls = AutoModelForVision2Seq if entry.model_type == "vlm" else AutoModelForCausalLM
        kw = dict(torch_dtype=self.cfg.hf_dtype, device_map=self.cfg.hf_device_map,
                  attn_implementation=self.cfg.hf_attn_impl,
                  trust_remote_code=self.cfg.hf_trust_remote_code)
        # compressed-tensors checkpoints auto-detect via config.json
        # GPTQ via auto-gptq integration if installed
        # AWQ via autoawq integration if installed
        # modelopt FP8 -> requires nvidia-modelopt installed; load via mtq.quantize() restore path
        self.model = cls.from_pretrained(entry.path, **kw)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(entry.path)
        if entry.model_type == "vlm":
            self.processor = AutoProcessor.from_pretrained(entry.path)

    def forward_logits(self, input_ids):
        with torch.no_grad():
            return self.model(input_ids.to(self.model.device)).logits

    def generate_with_logits(self, prompts, max_new_tokens):
        # use model.generate(..., output_scores=True, return_dict_in_generate=True)
        ...

    def measure_ttft_tpot(self, prompt, n=100):
        # transformers TextIteratorStreamer + threading to capture first-token time
        from transformers import TextIteratorStreamer
        import threading
        ttfts, tpots = [], []
        for _ in range(n):
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            t0 = time.perf_counter()
            th = threading.Thread(target=self.model.generate,
                kwargs=dict(**ids, max_new_tokens=128, streamer=streamer, do_sample=False))
            th.start()
            first = next(iter(streamer)); t_first = time.perf_counter()
            ttfts.append((t_first - t0) * 1000)
            tok_count = 1; t_prev = t_first
            for _ in streamer: tok_count += 1
            t_last = time.perf_counter()
            tpots.append(((t_last - t_first) / max(1, tok_count - 1)) * 1000)
            th.join()
        return {"ttft_ms_p50": median(ttfts), "ttft_ms_p95": p95(ttfts),
                "tpot_ms_p50": median(tpots), "tpot_ms_p95": p95(tpots)}

    def unload(self):
        del self.model
        torch.cuda.empty_cache(); gc.collect()
```

**Quant-format compatibility matrix (HF side):**

| Format | HF loader path | Notes |
|---|---|---|
| compressed-tensors (llmcompressor output) | native via `compressed-tensors` pip pkg | works for W4A16, W8A8, W8A8+smoothquant |
| GPTQ | `auto-gptq` integration (auto-detected by `config.json`) | install `auto-gptq` |
| AWQ | `autoawq` integration | install `autoawq` |
| bitsandbytes 4/8-bit | native `load_in_4bit=True` | not used here (different scheme) |
| modelopt FP8 | requires `nvidia-modelopt` + restore from state-dict path | `mtq.restore(model, ...)` after load |
| modelopt NVFP4 / MXFP4 | **HF cannot run** (no kernel) → vLLM only, skip HF quality eval | `entry.skip_hf_runtime = True` |
| FP16 baseline | trivial | reference |

Add `skip_hf_runtime: bool = False` to `BenchmarkModelEntry`. Auto-set by `detect.py` if format is HF-incompatible. Quality metrics for those models come from vLLM logprobs (limited).

#### 6.5.5 `VLLMRuntime` — perf eval owner

```python
class VLLMRuntime(RuntimeBase):
    name = "vllm"
    def load(self, entry):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=entry.path,
            quantization=entry.vllm_quantization,        # 'compressed-tensors' | 'modelopt' | None
            gpu_memory_utilization=entry.gpu_memory_utilization,
            max_model_len=entry.max_model_len,
            dtype=self.cfg.vllm_dtype,
            enforce_eager=self.cfg.vllm_enforce_eager,
            disable_log_stats=self.cfg.vllm_disable_log_stats,
            kv_cache_dtype=self.cfg.vllm_kv_cache_dtype,
            swap_space=self.cfg.vllm_swap_space_gb,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams

    def measure_ttft_tpot(self, prompt, n=100):
        # vLLM offline batch API gives per-request metrics in RequestOutput.metrics
        sp = self.SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
        ttfts, tpots = [], []
        for _ in range(n):
            out = self.llm.generate([prompt], sp, use_tqdm=False)[0]
            m = out.metrics
            ttfts.append((m.first_token_time - m.arrival_time) * 1000)
            decode_time = (m.finished_time - m.first_token_time)
            n_tok = len(out.outputs[0].token_ids)
            tpots.append((decode_time / max(1, n_tok - 1)) * 1000)
        return {"ttft_ms_p50": median(ttfts), "ttft_ms_p95": p95(ttfts),
                "tpot_ms_p50": median(tpots), "tpot_ms_p95": p95(tpots)}

    def measure_throughput(self, prompt, batch_sizes, output_len):
        rows = []
        for bs in batch_sizes:
            sp = self.SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True)
            t0 = time.perf_counter()
            outs = self.llm.generate([prompt] * bs, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            tot_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            rows.append({"batch": bs, "tput_tok_s": tot_tok / dt, "wall_s": dt})
        return rows

    def forward_logits(self, input_ids):
        # vLLM exposes logprobs (top-k), not full logits.
        # For logit-KL: use SamplingParams(logprobs=20, prompt_logprobs=20) and reconstruct.
        # Or raise NotImplementedError and force quality eval to HF.
        raise NotImplementedError("Use HFRuntime for logit-level metrics")

    def unload(self):
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del self.llm
        torch.cuda.empty_cache(); gc.collect()
```

#### 6.5.6 Runner orchestration — per-model both-runtimes loop

`runner.py` per model:

```python
def evaluate_model(entry, cfg, tracker):
    results = {"model": entry.dict(), "runtimes": {}}

    # ---- HF runtime: quality ----
    if cfg.runtime.hf and not entry.skip_hf_runtime:
        with run_in_subprocess() as proc:           # isolate so OOM/segfault doesn't kill perf run
            hf = HFRuntime(cfg.runtime)
            hf.load(entry)
            results["runtimes"]["hf"] = {
                "load_time_s": hf.last_load_s,
                "peak_vram_mb": hf.peak_vram_mb(),
                "quality": {
                    "ppl":         eval_ppl(hf, cfg.datasets) if "ppl" in cfg.metrics.quality_llm else None,
                    "mmlu_tiny":   eval_mmlu_tiny(hf, cfg.datasets) if "mmlu_tiny" in cfg.metrics.quality_llm else None,
                    "logit_kl":    eval_logit_kl(hf, baseline_logits) if "logit_kl" in cfg.metrics.quality_llm else None,
                    "token_agree": eval_token_agree(hf, baseline_outputs) if "token_agree" in cfg.metrics.quality_llm else None,
                    "ocr":         eval_ocr(hf, cfg.datasets) if entry.model_type == "vlm" else None,
                },
                # also measure HF-side TTFT/TPOT (compare to vLLM gap == kernel cost)
                "perf": hf.measure_ttft_tpot("Hello world.", n=20) if cfg.runtime.perf_runtime in ("hf", "both") else None,
            }
            hf.unload()

    # ---- vLLM runtime: perf ----
    if cfg.runtime.vllm:
        with run_in_subprocess() as proc:
            vl = VLLMRuntime(cfg.runtime)
            vl.load(entry)
            results["runtimes"]["vllm"] = {
                "load_time_s": vl.last_load_s,
                "peak_vram_mb": vl.peak_vram_mb(),
                "perf": {
                    "ttft_tpot":  vl.measure_ttft_tpot(STD_PROMPT, n=cfg.latency.num_requests),
                    "throughput": vl.measure_throughput(STD_PROMPT, cfg.latency.batch_sizes,
                                                       cfg.latency.output_lens[0]),
                    "ctx_sweep":  vl.measure_ctx_sweep(cfg.latency.ctx_sweep),
                },
                # cross-validate quality (small sample) for vLLM if HF was skipped
                "quality": (vl.eval_quality_via_logprobs(...) if entry.skip_hf_runtime else None),
            }
            vl.unload()

    # ---- cross-check parity ----
    if cfg.runtime.hf and cfg.runtime.vllm and not entry.skip_hf_runtime:
        # generate 20 prompts on both with greedy decoding, compare exact-match %
        results["runtime_parity"] = cross_check_generation(hf_results, vllm_results)
        # ttft/tpot delta = pure runtime overhead, not quant cost
        results["runtime_overhead"] = {
            "ttft_hf_vs_vllm_ms": hf_ttft - vllm_ttft,
            "tpot_hf_vs_vllm_ms": hf_tpot - vllm_tpot,
        }
    return results
```

**Critical: never load both runtimes simultaneously.** Each runs in its own subprocess; load → eval → unload → fork next. On 3060 (12GB) loading both = OOM. On MI300X (192GB) it works but corrupts NCCL/parallel state.

#### 6.5.7 Output JSON updates

Per-model JSON gains a `runtimes` block:

```jsonc
{
  "model": {...},
  "hardware": {...},
  "runtimes": {
    "hf": {
      "load_time_s": 18.2,
      "peak_vram_mb": 6800,
      "quality": {"ppl": 6.82, "mmlu_tiny_acc": 0.61, "logit_kl_mean": 0.018, "ocr": {...}},
      "perf":    {"ttft_ms_p50": 220.1, "tpot_ms_p50": 22.3}
    },
    "vllm": {
      "load_time_s": 24.5,
      "peak_vram_mb": 9100,
      "perf": {"ttft_ms_p50": 88.1, "tpot_ms_p50": 6.2, "throughput_tok_s": [...], "ctx_sweep": [...]},
      "quality": null
    }
  },
  "runtime_parity": {"exact_match_pct": 98.5, "token_agree_pct": 99.2},
  "runtime_overhead": {"ttft_hf_vs_vllm_ms": 132.0, "tpot_hf_vs_vllm_ms": 16.1},
  "delta_vs_baseline": {...},
  "errors": [],
  "timestamp": "..."
}
```

#### 6.5.8 New plots from dual-runtime data

| Plot | Insight |
|---|---|
| **HF-vs-vLLM TTFT bar** per model | runtime overhead = kernel maturity cost |
| **HF-vs-vLLM TPOT bar** | same, decode phase |
| **Runtime-parity heatmap** | exact-match % across (model × scheme) — sanity gate |
| **Quality-from-HF vs perf-from-vLLM scatter** | one dot per model; lower-right corner = fast + accurate |

Add to `plots.py` dispatch.

#### 6.5.9 CLI flags

```bash
# Both runtimes (default)
python benchmark.py --queue

# vLLM only (skip HF — faster, no quality)
python benchmark.py --queue --runtime vllm

# HF only (perf reference)
python benchmark.py --queue --runtime hf

# Skip a single model's HF eval (auto-detected for NVFP4)
# via YAML: skip_hf_runtime: true
```

### 6.6 Hardware detection

```python
def detect_hw():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)           # "NVIDIA GeForce RTX 3060" or "AMD Instinct MI300X"
        props = torch.cuda.get_device_properties(0)
        major, minor = props.major, props.minor
        if "AMD" in name or "Instinct" in name:
            return {"vendor": "amd", "name": name, "arch": f"gfx{major}{minor}{...}",
                    "runtime": "rocm", "vram_gb": props.total_memory / 1e9}
        else:
            return {"vendor": "nvidia", "name": name, "arch": f"sm_{major}{minor}",
                    "runtime": "cuda", "vram_gb": props.total_memory / 1e9}
    return {"vendor": "cpu"}
```

`skip_on` honored: if `"sm_86" in skip_on` and detected arch matches → mark `skipped`, continue.

---

## 7. Plots (matplotlib local + W&B / MLflow upload)

All plots one-shot from `aggregate.py` reading `results/{run_id}/summary.csv`. No interactive backends. Save 300dpi PNG locally **and** push to W&B (`wandb.Image`) + MLflow (`mlflow.log_figure`). Matplotlib `Figure` object goes to both — no double-rendering.

```python
def emit_plot(fig, name, tracker):
    fig.savefig(f"results/{run_id}/plots/{name}.png", dpi=300, bbox_inches="tight")
    tracker.log_plot(name, fig)         # fanout to wandb + mlflow
    plt.close(fig)
```

W&B treats every plot as both an `Image` and (where possible) a native `wandb.plot.line/bar/scatter` so users can interactively re-filter in the dashboard. Pareto plots use `wandb.plot.scatter` for hover-tooltips on model names.

| Plot | X | Y | Style | Use |
|---|---|---|---|---|
| **PPL bar** | model | PPL | grouped bar, color by backend, baseline horizontal line | quality at-a-glance |
| **CER bar** | model | CER (%) | same | OCR quality |
| **Throughput vs batch** | batch_size | tok/s | line per model | scaling behavior |
| **TTFT cdf** | latency_ms | cumulative % | line per model | tail latency |
| **VRAM vs ctx** | ctx_len | peak_vram_mb | line per model | KV scaling |
| **Pareto: quality vs latency** | ttft_ms | PPL | scatter, lower-left better, baseline marked | tradeoff picker |
| **Pareto: quality vs size** | disk_mb | PPL | scatter | compression tradeoff |
| **Speedup matrix** | scheme | model | heatmap of speedup vs FP16 | cross-backend perf |
| **Quality-preservation matrix** | scheme | model | heatmap of `1 - ΔPPL/PPL_fp16` | cross-backend quality |
| **Ctx sweep tput** | ctx | tok/s | line per model+scheme, log-x | long-context regression |
| **Compression-vs-quality** | compression_ratio | quality_pres | scatter, ideal corner top-right | best-of-both ranker |

All plots reused for OCR by swapping PPL→CER.

---

## 7b. Experiment Tracking — Multi-Backend Fanout

### 7b.1 Why three trackers

| Tracker | Strength | Role here | Public URL? |
|---|---|---|---|
| **W&B** | best plots + interactive dashboards + reports + artifact versioning + group/sweep views | **primary** — all metrics, plots, model artifacts, summary tables | ✅ public project — shareable run/report links |
| **Langfuse** | LLM-trace native: prompt → response → score per call | **OCR per-sample traces** — every image+prompt+pred+CER score, browsable, filterable by score | ✅ shareable trace/session URLs |
| **MLflow** | open standard, model registry, cross-tool metric source-of-truth | **archive + registry** — metric truth + artifact backup + optional model registration | ⚠️ only if user runs an MLflow server; local `file:` URI by default |

Goals served:
- W&B = humans look at plots and compare runs.
- Langfuse = debug *why* one quant scheme produces bad OCR on specific samples.
- MLflow = machine-readable history, reproducibility, model registry hooks for CI/deploy.

### 7b.2 `TrackerBase` ABC

```python
class TrackerBase(ABC):
    @abstractmethod
    def start_run(self, run_name: str, config: dict, tags: list[str]) -> str: ...
    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    @abstractmethod
    def log_plot(self, name: str, fig: "matplotlib.figure.Figure") -> None: ...
    @abstractmethod
    def log_artifact(self, path: str, artifact_type: str = "file") -> None: ...
    @abstractmethod
    def log_table(self, name: str, df: "pandas.DataFrame") -> None: ...
    @abstractmethod
    def log_trace(self, name: str, inputs: dict, outputs: dict, scores: dict) -> None: ...
    @abstractmethod
    def get_run_url(self) -> str | None: ...
    @abstractmethod
    def finish(self) -> None: ...
```

### 7b.3 `CompositeTracker` — fanout

```python
class CompositeTracker(TrackerBase):
    def __init__(self, trackers: list[TrackerBase]):
        self.trackers = trackers
    def log_metric(self, k, v, step=None):
        for t in self.trackers:
            try: t.log_metric(k, v, step)
            except Exception as e: log.warning("tracker %s failed: %s", t, e)
    # ... same pattern for every method — never let one tracker crash the run
```

Per-tracker exception isolation is mandatory. W&B network blip should not kill MLflow logging or local PNG save.

### 7b.4 W&B implementation specifics

- Run init: `wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_name, config=full_config_dict, tags=cfg.wandb_tags + [hardware.vendor, hardware.arch], reinit=True)`.
- **Per-model = nested group:** `wandb.init(group=run_name, job_type=model_name)`. Each model is its own W&B run, all grouped under the sweep — dashboard shows them side-by-side automatically.
- Metrics logged with `step` = eval-stage index (so the run timeline shows: load → ppl → mmlu → ocr → latency).
- **Plots**: `wandb.log({f"plots/{name}": wandb.Image(fig)})` + when possible, `wandb.plot.bar(wandb.Table(...), ...)` for interactive versions.
- **Summary table**: `wandb.log({"summary": wandb.Table(dataframe=summary_df)})`.
- **Artifacts**: quantized model dirs uploaded only if `< 5 GB` (skip large LLMs to save bandwidth; just log path). Config YAML + per-model JSON always uploaded.
- **OCR per-sample**: `wandb.Table(columns=["image", "ref", "pred", "cer", "wer"])` — top-50 worst samples by CER, with `wandb.Image(pil_image)`.
- **Cross-run report**: after sweep, `aggregate.py` calls `wandb.Api()` to auto-generate a Report with: leaderboard table, key plots, hardware comparison panel. Public URL printed to stdout.

### 7b.5 Langfuse implementation specifics

Langfuse models each OCR prediction as a `trace` → `generation` event:

```python
from langfuse import Langfuse
lf = Langfuse(public_key=..., secret_key=..., host=...)

session = lf.create_session(name=f"{run_name}/{model_name}/ocr")
for sample in ocr_samples:
    trace = lf.trace(name="ocr_predict", session_id=session.id, input={"image_id": sample.id})
    gen = trace.generation(
        name="vlm_generate",
        model=model_name,
        input={"prompt": prompt, "image_url": sample.url_or_b64},
        output={"text": pred},
        metadata={"quantization": scheme, "backend": backend},
    )
    trace.score(name="cer", value=cer_score, comment=f"ref: {ref[:100]}")
    trace.score(name="wer", value=wer_score)
    trace.score(name="exact_match", value=int(pred == ref))
```

Only enabled when `tracking.langfuse_only_ocr=True` (default) — PPL eval generates no Langfuse calls. Cost: ~500 traces per VLM run.

Session URL added to `tracker_urls.json` and printed to stdout. Public sharing toggled via Langfuse UI per-project setting (not API).

### 7b.6 MLflow implementation specifics

- URI: `mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)`. Default `file:./mlruns` — works offline, no server needed.
- Experiment: `mlflow.set_experiment(cfg.mlflow_experiment)`.
- One MLflow run per model: `with mlflow.start_run(run_name=f"{run_name}/{model_name}", nested=False):`.
- `mlflow.log_params(config_flat)`, `mlflow.log_metric("ppl", v)`, `mlflow.log_metric("cer", v)`, etc.
- `mlflow.log_artifact("plots/")` after all plots written — bulk upload.
- `mlflow.log_figure(fig, "plots/ppl_bar.png")` for in-line plot logging.
- **Model registry** (opt-in via `mlflow_register_model=True`): `mlflow.transformers.log_model(...)` — wraps the quantized model so it's pullable from registry by name+version. For HF compressed-tensors models, register the directory as artifact + add a `pyfunc` loader that wraps `vllm.LLM(...)`.
- `mlflow.set_tag("hw.vendor", ...)`, `mlflow.set_tag("hw.arch", ...)`, `mlflow.set_tag("quant.backend", ...)` — enables filter/group in MLflow UI.

### 7b.7 Credentials & offline fallback

- Each tracker checks its env var (e.g., `WANDB_API_KEY`). If missing AND `tracking.offline_mode=False` → log warning, swap that tracker for `NoOpTracker` (other trackers still active).
- If ALL trackers fail/missing → only local PNG/JSON/CSV produced. Run succeeds. Print summary URL list at end (may be empty).
- `--no-tracking` CLI flag → forces `CompositeTracker([NoOpTracker()])`.

### 7b.8 Run-URL bookkeeping

`results/{run_id}/tracker_urls.json`:
```jsonc
{
  "wandb": {
    "sweep_url": "https://wandb.ai/<entity>/triplequant-vlm/groups/<run_id>",
    "report_url": "https://wandb.ai/<entity>/triplequant-vlm/reports/...",
    "per_model": {"tinyllama-w4a16": "https://wandb.ai/.../runs/<id>", ...}
  },
  "langfuse": {
    "session_urls": {"qwen2.5-vl-3b-awq": "https://cloud.langfuse.com/project/<id>/sessions/<sid>"}
  },
  "mlflow": {
    "experiment_id": "12",
    "experiment_uri": "http://mlflow.local/#/experiments/12",
    "per_model": {"tinyllama-w4a16": "12/<run_uuid>"}
  }
}
```
Printed to stdout at run end. Also appended to a top-level `results/_index.md` so user has one file linking every sweep ever run.

### 7b.9 Dual-runtime tagging in trackers

Every metric tagged with `runtime ∈ {hf, vllm}` so the same model produces two metric series. Plot panels filter on `runtime`.

- W&B: `wandb.log({"hf/ppl": v, "vllm/ttft_p50": v})` — namespace by runtime prefix; panel groups auto-create.
- MLflow: `mlflow.log_metric("hf.ppl", v)` + tag `runtime=hf` for cross-run filter.
- Langfuse: trace tag `runtime=hf` (OCR predictions always HF — vLLM doesn't expose per-sample VLM trace cleanly).

Per-runtime sub-run pattern (W&B):
```python
with wandb.init(group=model_name, job_type="hf", reinit=True) as hf_run:
    log_quality_metrics(hf_run)
with wandb.init(group=model_name, job_type="vllm", reinit=True) as vllm_run:
    log_perf_metrics(vllm_run)
```
Group view shows them side-by-side; charts can split or merge on `job_type`.

### 7b.10 Quantize-time tracking (lightweight)

`quantize.py` also opens a tracker context — short run logging:
- params: full QuantizeConfig
- metrics: calibration loss (if exposed), wall time per phase, final model size MB
- artifact: the YAML config + final `config.json` of saved model
- tag: `step="quantize"` (vs `step="benchmark"`)

This gives a single W&B group per model lifecycle: quantize run + benchmark run linked by a shared `group=<model_name>` tag.

---

## 8. Crash-Safety & Logging

- Each model evaluated inside a subprocess (`multiprocessing.Process` or `subprocess.run`). CUDA OOM in model N doesn't poison N+1.
- Result JSON written **per-eval-stage** not per-model — if latency crashes, quality results already saved.
- Per-model log file: `results/{run_id}/logs/{model_name}.log`. INFO to stdout, DEBUG to file.
- `--resume`: scan `results/{run_id}/`, skip models with complete JSON, retry partial/missing.
- Hardware fingerprint written once per run; mismatched fingerprint on `--resume` aborts with clear error.

---

## 9. Cross-Hardware Comparison Mode

`benchmark.py --merge-runs results/run_3060_A results/run_mi300x_B --out results/cross_hw_X`

Produces:
- `compare.csv` — per (model, metric) rows for each hardware
- `compare_plots/`:
  - `tput_3060_vs_mi300x.png` — bar pair per model
  - `vram_efficiency.png` — `tput / vram_used` per model per hw
  - `quality_invariance.png` — verify PPL/CER identical across hw (sanity — should be ±0.05)

Quality MUST match across hardware (same weights). If `|ΔPPL_hw1_hw2| > 0.1`, flag potential numerics divergence (e.g., FP8 emulation on 3060 vs native MI300X).

---

## 10. Implementation Timeline (overlays `plan.md` Week 1)

| Day | Task | Deliverable |
|---|---|---|
| Day 1 (Fri 5/22) | `BenchmarkConfig` + `MetricsConfig` + `BenchmarkModelEntry` schemas; fix `loader.py` import. `hardware.py` detect. | Pydantic round-trip test. |
| Day 2 | `eval_llm.py::compute_ppl` (wikitext-2). `eval_memory.py` disk + VRAM helpers. | PPL of TinyLlama matches HF reference within ±0.05. |
| Day 3 | `eval_ocr.py` CER/WER on LaTeX_OCR. Normalizer for LaTeX tokens. | CER computed on Qwen2.5-VL-3B FP16 baseline. |
| Day 4 | `eval_latency.py` via `vllm.LLM` — TTFT, TPOT, throughput. Batch sweep. Ctx sweep. | Latency numbers for TinyLlama on 3060. |
| Day 5 | `runner.py` orchestrator + crash-safe subprocess. Per-model JSON writer. Auto-detect `vllm_quantization`. | End-to-end one model run + JSON dump. |
| Day 6 | `enqueue_for_benchmark()` hook in both quantizers. `benchmark_queue.yaml` read/write. CLI `--queue`/`--clear-queue`/`--retry-failed`. | Quantize TinyLlama → auto-appears in queue → benchmark consumes. |
| Day 7 | `plots.py` 6 core plots. `aggregate.py` CSV + plot dispatch. **`src/tracking/{base,composite,wandb_tracker,mlflow_tracker,langfuse_tracker,noop}.py` — fanout wiring.** MMLU-tiny + logit-KL diagnostics. | Run 3-model sweep on 3060 → local PNG + W&B run URL + MLflow run + Langfuse OCR session, all reachable from `tracker_urls.json`. |
| Day 7b (overflow) | W&B auto-Report generator in `aggregate.py`. Quantize-time tracker context in `quantize.py`. `--no-tracking` flag. Credential graceful-degrade. | Public W&B Report URL printed; offline run still works. |
| Day 8+ | (handoff to TurboQuant week) MI300X parity run when hardware available; cross-hw merge plot script. | Cross-HW table once both runs collected; W&B group spans both hardwares for direct compare. |

Total: **7 days dev within Week 1.** Tight but feasible — most evaluators are 30–80 LOC each.

---

## 11. Test Matrix (first full sweep target)

| # | Model | Backend | Scheme | 3060 | MI300X |
|---|---|---|---|---|---|
| 1 | TinyLlama-1.1B FP16 (baseline) | — | FP16 | ✅ | ✅ |
| 2 | TinyLlama-1.1B | llmcompressor | GPTQ-W4A16 | ✅ | ✅ |
| 3 | TinyLlama-1.1B | llmcompressor | GPTQ-W8A8 + SmoothQuant | ✅ | ✅ |
| 4 | TinyLlama-1.1B | modelopt | FP8 | ⚠️ emulated | ✅ native |
| 5 | Qwen2.5-1.5B | llmcompressor | AWQ-W4A16 | ✅ | ✅ |
| 6 | Qwen2.5-VL-3B FP16 | — | FP16 | ✅ | ✅ |
| 7 | Qwen2.5-VL-3B | llmcompressor | AWQ-W4A16 | ✅ | ✅ |
| 8 | Qwen2.5-VL-3B | llmcompressor | GPTQ-W4A16 | ✅ | ✅ |
| 9 | SmolVLM2-2.2B | llmcompressor | GPTQ-W4A16 | ✅ | ✅ |
| 10 | Llama-3-8B FP16 | — | FP16 | ⚠️ tight | ✅ |
| 11 | Llama-3-8B | llmcompressor | GPTQ-W4A16 | ✅ | ✅ |
| 12 | Llama-3-8B | modelopt | FP8 | ❌ skip | ✅ |
| 13 | Qwen2.5-VL-7B | llmcompressor | AWQ-W4A16 | ❌ skip | ✅ |
| 14 | Mixtral-8x7B | modelopt | FP8 | ❌ skip | ✅ |

`skip_on: ["sm_86"]` on rows 12–14. Rows 1–11 = parity table.

---

## 12. Open Questions

1. **MI300X access** — when does hardware come online? Drives whether week-1 ships single-hw or dual-hw report. Default assume 3060-only at first, MI300X added when available.
2. **Concurrent batch >32 on 3060?** Likely OOM for 8B models. Skip or just early-stop on OOM and record max-batch-supported as a metric itself?
3. **MMLU full vs tiny?** Full = 14K q × N choices, ~hours per model. Tiny (500q) is what fits in benchmark loop. Acceptable?
4. **vLLM version pin?** ROCm support moves fast — pin one for 3060 (CUDA) and one for MI300X (ROCm) in `req.txt` to avoid silent regressions.
5. **Plot color scheme** — auto-distinct per model, or fixed per backend (llmcompressor=blue, modelopt=orange, fp16=grey)? Latter is cleaner for cross-backend reading. Same palette mirrored to W&B run colors via `wandb.init(settings=wandb.Settings(...))`.
6. **W&B project — public or private?** Default public (shareable). Need W&B entity name + API key in env.
7. **Langfuse hosting** — cloud (`cloud.langfuse.com`, public traces possible) or self-host? Cloud free tier covers ~50K events/mo — enough for OCR sweeps. Self-host if data-sensitivity matters.
8. **MLflow server** — local `file:./mlruns` enough for solo work; need a server (`mlflow server --backend-store-uri ... --artifacts-destination s3://...`) for team sharing. Default local; user upgrades when ready.
9. **OCR ground-truth normalization** — LaTeX_OCR has known annotation noise. Use exact-string CER, or apply known-issue filter (drop samples with `\frac{}{}` empty braces etc.)?
10. **Per-sample upload volume** — `log_per_sample_predictions=True` pushes every OCR pred to W&B Table + Langfuse trace. ~500 samples × N models can hit free-tier limits. Default off; on for "release" runs.

---

## 12b. Tracker Integration Deep-Dive

End-to-end concrete patterns for each tracker. All three live behind `TrackerBase`; this section shows the actual SDK calls per backend, env var handling, run lifecycle, artifact bundling, failure isolation, and how the **dual runtime** (vLLM + HF) data feeds into each.

### 12b.1 Common run lifecycle

```
benchmark.py main()
  ├─ load BenchmarkConfig
  ├─ resolve secrets from env (WANDB_API_KEY, LANGFUSE_*, MLFLOW_TRACKING_*)
  ├─ build CompositeTracker (per cfg.tracking.enabled)
  ├─ tracker.start_run(run_name=cfg.run_name, config=cfg.dict(), tags=[...])
  ├─ for entry in queue:
  │    ├─ tracker.start_model(entry.name)        # opens per-model sub-run
  │    ├─ for runtime in (hf, vllm):
  │    │    ├─ for stage in (load, quality, perf, memory):
  │    │    │    ├─ run stage
  │    │    │    ├─ tracker.log_metrics({"<runtime>/<stage>/<k>": v, ...}, step=stage_idx)
  │    │    │    └─ tracker.log_artifact(stage_artifact_path) if any
  │    ├─ tracker.log_table("per_sample_ocr", df)
  │    ├─ tracker.log_plot("ppl_bar", fig)
  │    └─ tracker.finish_model()
  ├─ aggregate.py: build summary CSV + plots → tracker.log_table/log_plot at sweep level
  ├─ tracker.finalize_report()                   # W&B Report, MLflow comparison URL
  └─ tracker.finish()
```

Sub-runs nested under `group=run_name` so the dashboard shows one parent + N children (one per model + runtime).

### 12b.2 W&B — full integration (primary)

#### 12b.2.1 Auth + init

```python
import os, wandb

def init_wandb(cfg, run_name, hw, group=None, job_type=None):
    if os.environ.get(cfg.tracking.wandb_api_key_env) is None:
        if cfg.tracking.offline_mode:
            os.environ["WANDB_MODE"] = "offline"
        else:
            log.warning("WANDB_API_KEY not set; falling back to NoOp")
            return None
    return wandb.init(
        project=cfg.tracking.wandb_project,
        entity=cfg.tracking.wandb_entity,
        name=run_name,
        group=group or cfg.run_name,
        job_type=job_type or "sweep",
        config=cfg.dict(),
        tags=cfg.tracking.wandb_tags + [hw["vendor"], hw["arch"], "triplequant"],
        reinit=True,
        settings=wandb.Settings(start_method="thread",
                                console="redirect",
                                disable_stats=False),
    )
```

#### 12b.2.2 Metric logging — namespace by runtime + stage

```python
def log_metrics_wandb(run, runtime, stage, metrics, step):
    payload = {f"{runtime}/{stage}/{k}": v for k, v in metrics.items()}
    payload["_step_stage"] = stage
    run.log(payload, step=step)
```

Naming convention everywhere:
- `hf/quality/ppl`, `hf/quality/mmlu_tiny_acc`, `hf/quality/ocr/cer`
- `vllm/perf/ttft_ms_p50`, `vllm/perf/throughput/bs8_tok_s`
- `*/memory/peak_vram_mb`, `*/memory/disk_mb`
- `delta/ppl_vs_fp16`, `delta/compression_ratio`, `delta/speedup`
- `parity/exact_match_pct` (only when both runtimes ran)

Dashboards filter on prefix; one config can produce all panels.

#### 12b.2.3 Plots — image + native interactive

```python
def log_plot_wandb(run, name, fig, df=None, kind=None):
    run.log({f"plots/{name}": wandb.Image(fig)})
    if df is not None and kind == "bar":
        tbl = wandb.Table(dataframe=df)
        run.log({f"interactive/{name}": wandb.plot.bar(tbl, "model", "value", title=name)})
    elif df is not None and kind == "scatter":
        tbl = wandb.Table(dataframe=df)
        run.log({f"interactive/{name}": wandb.plot.scatter(tbl, "x", "y", title=name)})
```

Pareto plots use `wandb.plot.scatter` with model-name tooltips; bar plots use `wandb.plot.bar`. Static PNG always logged in parallel.

#### 12b.2.4 Artifacts — quantized model + configs + per-model JSON

```python
def log_artifact_wandb(run, path, name, artifact_type="model", max_size_gb=5):
    p = Path(path)
    if p.is_file():
        size_gb = p.stat().st_size / 1e9
    else:
        size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
    if size_gb > max_size_gb:
        log.info(f"Skipping artifact {name} ({size_gb:.1f} GB > {max_size_gb} GB cap)")
        run.config[f"artifact_skipped/{name}"] = {"size_gb": size_gb, "path": str(p)}
        return
    art = wandb.Artifact(name=name, type=artifact_type,
                         metadata={"size_gb": size_gb, "sha": sha256_short(p)})
    if p.is_file():
        art.add_file(str(p))
    else:
        art.add_dir(str(p))
    run.log_artifact(art, aliases=["latest", run.name])
```

What goes up:
- `config.yaml` (the BenchmarkConfig) — always.
- `results/{run_id}/{model}.json` — always.
- `results/{run_id}/summary.csv` — at sweep end.
- Quantized model dir — only if `< 5 GB` AND `cfg.tracking.log_artifacts=True`. For larger models log a `wandb.config["model_path"]` pointer instead.
- HuggingFace dataset snapshot SHA — for reproducibility.

#### 12b.2.5 Tables — summary + per-sample OCR

```python
# Sweep summary (one row per model × runtime)
summary_df = pd.read_csv(f"results/{run_id}/summary.csv")
run.log({"summary_table": wandb.Table(dataframe=summary_df)})

# Per-sample OCR (top-50 worst, with images)
worst = ocr_df.sort_values("cer", ascending=False).head(50)
img_table = wandb.Table(columns=["sample_id", "image", "ref", "pred", "cer", "wer", "scheme"])
for _, r in worst.iterrows():
    img_table.add_data(r.id, wandb.Image(load_img(r.image_path)),
                       r.ref, r.pred, r.cer, r.wer, r.scheme)
run.log({"ocr/worst_50": img_table})
```

W&B Tables natively render images + sortable columns. Critical for "why did this scheme break formula X."

#### 12b.2.6 Auto-Report generation

```python
from wandb.apis.reports import Report, PanelGrid, LinePlot, BarPlot, ScatterPlot, RunComparer

def generate_report(api, entity, project, group, hw):
    report = Report(
        project=project, entity=entity,
        title=f"TripleQuant Sweep — {group} ({hw['name']})",
        description=f"Auto-generated. Hardware: {hw}. Runtimes: hf + vllm.",
        blocks=[
            PanelGrid(panels=[
                BarPlot(title="PPL vs FP16",       x="model", y="hf/quality/ppl"),
                BarPlot(title="CER (OCR)",         x="model", y="hf/quality/ocr/cer"),
                LinePlot(title="Throughput vs batch", x="batch", y="vllm/perf/throughput/tok_s"),
                LinePlot(title="VRAM vs ctx",      x="ctx",   y="vllm/memory/peak_vram_mb"),
                ScatterPlot(title="Pareto: PPL vs TTFT", x="vllm/perf/ttft_ms_p50", y="hf/quality/ppl"),
                ScatterPlot(title="Compression vs Quality",
                            x="delta/compression_ratio", y="delta/quality_pres"),
                BarPlot(title="HF vs vLLM TTFT overhead", x="model", y="runtime_overhead/ttft_ms"),
            ]),
            RunComparer(runs_per_page=20),
        ],
    )
    return report.save().url
```

URL printed to stdout + saved into `tracker_urls.json`. One sweep = one shareable URL.

#### 12b.2.7 Error isolation

```python
class WandbTracker(TrackerBase):
    def _safe(self, fn, *a, **kw):
        if self.run is None: return
        try: fn(*a, **kw)
        except wandb.errors.CommError as e:
            log.warning("wandb network err (continuing): %s", e)
        except Exception as e:
            log.warning("wandb err (continuing): %s", e)
    def log_metric(self, k, v, step=None): self._safe(self.run.log, {k: v}, step=step)
```

W&B network blip never crashes a benchmark run. Sweep continues; final JSON is the source of truth.

### 12b.3 Langfuse — full integration (OCR per-sample LLM-trace)

#### 12b.3.1 Auth + client

```python
from langfuse import Langfuse, observe
from langfuse.decorators import langfuse_context

def init_langfuse(cfg):
    pk = os.environ.get(cfg.tracking.langfuse_public_key_env)
    sk = os.environ.get(cfg.tracking.langfuse_secret_key_env)
    if not (pk and sk):
        if cfg.tracking.offline_mode:
            return None
        log.warning("LANGFUSE creds missing; NoOp")
        return None
    return Langfuse(public_key=pk, secret_key=sk,
                    host=cfg.tracking.langfuse_host,
                    flush_at=20, flush_interval=5.0)   # batch send
```

#### 12b.3.2 Session per (run × model × runtime)

```python
def open_ocr_session(lf, run_name, model_name, runtime="hf"):
    return lf.create_session(
        name=f"{run_name}/{model_name}/{runtime}/ocr",
        metadata={"runtime": runtime, "model": model_name, "run_id": run_name},
    )
```

#### 12b.3.3 Per-sample trace — full prompt+image+pred+scores

```python
def trace_ocr_sample(lf, session_id, model_name, runtime, scheme, backend,
                     sample, prompt, pred, cer, wer, exact_match, gen_meta):
    trace = lf.trace(
        name="ocr_predict",
        session_id=session_id,
        input={"image_id": sample.id, "image_b64": sample.image_b64[:5000] + "..."},
        tags=[runtime, scheme, backend, sample.dataset],
        metadata={"ref_len": len(sample.ref), "ref_preview": sample.ref[:200]},
    )
    gen = trace.generation(
        name="vlm_generate",
        model=model_name,
        model_parameters={"max_new_tokens": gen_meta["max_new_tokens"],
                          "temperature": gen_meta["temperature"]},
        input={"prompt": prompt, "image_ref": sample.image_url_or_path},
        output={"text": pred},
        usage={"input_tokens": gen_meta["prompt_tokens"],
               "output_tokens": gen_meta["completion_tokens"]},
        metadata={"quantization": scheme, "backend": backend,
                  "runtime": runtime, "latency_ms": gen_meta["latency_ms"]},
    )
    trace.score(name="cer", value=cer, comment=f"ref={sample.ref[:80]}…")
    trace.score(name="wer", value=wer)
    trace.score(name="exact_match", value=int(exact_match), data_type="BOOLEAN")
    if cer > 0.5:
        trace.score(name="failure_severity", value="high", data_type="CATEGORICAL")
```

Browse-by-score in Langfuse UI: filter `score(cer) > 0.3, tag=W4A16`. Click into individual trace → see prompt + image + pred + ref side-by-side. This is what makes a scheme failure debuggable.

#### 12b.3.4 Aggregation tracking — sweep summary

After OCR eval:
```python
sweep_trace = lf.trace(
    name="ocr_sweep_summary",
    session_id=session_id,
    output={"mean_cer": mean_cer, "median_cer": median_cer, "n_samples": n},
    metadata={"scheme": scheme, "model": model_name},
)
sweep_trace.score(name="mean_cer", value=mean_cer)
sweep_trace.score(name="p95_cer", value=p95_cer)
```

#### 12b.3.5 Flush + finish

```python
def finish_langfuse(lf):
    if lf is None: return
    lf.flush()                        # drain queue
    lf.shutdown()
```

Always called in `finally:` block — Langfuse uses background flush; missing shutdown = lost traces.

#### 12b.3.6 Public sharing

Langfuse traces public-shareable per-project setting. Toggle once via Langfuse UI (`Project → Settings → Public Traces`). After that, every trace/session URL is shareable without auth.

URL format saved to `tracker_urls.json`:
```
https://cloud.langfuse.com/project/<pid>/sessions/<sid>
```

#### 12b.3.7 Why-not-vLLM

vLLM `RequestOutput` has prompt + completion but doesn't expose generation in a way that's easy to trace per sample without a server wrapper. For OCR, run via HF runtime (which already owns quality). For vLLM-only quant schemes (NVFP4), wrap with a thin `OpenAI-compatible` API + the Langfuse OpenAI integration (`from langfuse.openai import openai`) — drop-in, all calls auto-traced.

### 12b.4 MLflow — full integration (archive + registry)

#### 12b.4.1 Tracking URI

```python
import mlflow
def init_mlflow(cfg):
    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)   # file:./mlruns or http://...
    mlflow.set_experiment(cfg.tracking.mlflow_experiment)
```

Local file backend works offline (default). Team/CI: spin a server:
```bash
mlflow server --backend-store-uri postgresql://... \
              --artifacts-destination s3://bucket/triplequant/ \
              --host 0.0.0.0 --port 5000
```
Then `mlflow_tracking_uri: http://mlflow-host:5000` in YAML.

#### 12b.4.2 Run hierarchy

```python
# Parent: the whole sweep
with mlflow.start_run(run_name=cfg.run_name, tags={"step": "benchmark", "sweep": "true"}) as parent:
    mlflow.log_params(flatten_dict(cfg.dict()))
    mlflow.log_dict(cfg.dict(), "config.yaml")

    for entry in queue:
        # Child per (model × runtime)
        for runtime in active_runtimes:
            with mlflow.start_run(run_name=f"{entry.name}/{runtime}", nested=True) as child:
                mlflow.set_tags({
                    "model": entry.name,
                    "runtime": runtime,
                    "backend": entry.backend_hint,
                    "scheme": detect_scheme(entry),
                    "hw.vendor": hw["vendor"],
                    "hw.arch": hw["arch"],
                })
                log_stage_metrics(child, runtime, results)
                mlflow.log_artifact(f"results/{run_id}/{entry.name}.json")
```

Nested = MLflow UI shows tree (sweep → model → runtime).

#### 12b.4.3 Metric logging — pure scalar + step

```python
def log_metrics_mlflow(metrics_dict, step=None):
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)) and not math.isnan(v):
            mlflow.log_metric(k.replace("/", "."), float(v), step=step)
        elif isinstance(v, list):
            for i, vi in enumerate(v):
                mlflow.log_metric(f"{k}.{i}", float(vi), step=step)
```

MLflow doesn't allow `/` in keys → use `.`. So W&B `hf/quality/ppl` becomes MLflow `hf.quality.ppl`. Same data, both views work.

#### 12b.4.4 Figure + artifact logging

```python
mlflow.log_figure(fig, f"plots/{name}.png")                 # direct from matplotlib
mlflow.log_artifact(f"results/{run_id}/plots/", "plots")    # bulk upload dir
mlflow.log_artifact(f"results/{run_id}/{model}.json", "raw")
mlflow.log_artifact(f"results/{run_id}/summary.csv",  "summary")
mlflow.log_text(json.dumps(hw, indent=2), "hardware.json")
```

#### 12b.4.5 Model registry (opt-in)

```python
def register_quantized_model(entry, mlflow_run, version_tag=None):
    if not cfg.tracking.mlflow_register_model: return
    # For HF compressed-tensors, register the *directory* as a pyfunc.
    import mlflow.pyfunc
    class CompressedTensorsModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from vllm import LLM
            self.llm = LLM(model=context.artifacts["model_dir"],
                           quantization=context.model_config["quantization"])
        def predict(self, context, model_input):
            return [o.outputs[0].text for o in
                    self.llm.generate(model_input.tolist(),
                        SamplingParams(max_tokens=128))]
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CompressedTensorsModel(),
        artifacts={"model_dir": entry.path},
        registered_model_name=f"triplequant/{entry.name}",
        metadata={"quantization": entry.vllm_quantization,
                  "backend": entry.backend_hint,
                  "version_tag": version_tag or "latest"},
        pip_requirements=["vllm>=0.14", "torch>=2.4", "compressed-tensors"],
    )
```

Result: each accepted scheme appears in MLflow registry as `triplequant/<model_name>`, version-bumped per re-run, pullable from CI by name+version. Off by default; flip on for "release" runs.

#### 12b.4.6 Tags drive UI filtering

```python
mlflow.set_tags({
    "step": "benchmark",            # 'quantize' | 'benchmark'
    "runtime": "vllm",              # 'hf' | 'vllm'
    "model_family": "qwen2.5-vl",
    "scheme": "awq-w4a16",
    "backend": "llm_compressor",
    "hw.vendor": "nvidia",
    "hw.arch": "sm_86",
    "ctx_max": "8192",
    "is_baseline": "false",
})
```

MLflow UI: filter `tags.scheme = 'awq-w4a16' AND tags.runtime = 'vllm'` → cross-model comparison view. URL bookmarkable.

#### 12b.4.7 Comparison URL

```python
def mlflow_comparison_url(experiment_id, run_ids):
    base = mlflow.get_tracking_uri().replace("file:", "http://localhost:5000")  # if server
    return f"{base}/#/experiments/{experiment_id}/compare-runs?runs={','.join(run_ids)}"
```

For local `file:./mlruns`, comparison is local-UI only (`mlflow ui --port 5000`). Document in run output.

### 12b.5 Cross-tracker bookkeeping

Single source `tracker_urls.json` updated at end of each model:
```jsonc
{
  "run_id": "2026-05-29T14-21-33Z_3060_full_sweep",
  "wandb": {
    "sweep_url": "https://wandb.ai/myorg/triplequant-vlm/groups/...",
    "report_url": "https://wandb.ai/myorg/triplequant-vlm/reports/...",
    "per_model_per_runtime": {
      "tinyllama-w4a16/hf":   "https://wandb.ai/.../runs/abc",
      "tinyllama-w4a16/vllm": "https://wandb.ai/.../runs/abd"
    }
  },
  "langfuse": {
    "host": "https://cloud.langfuse.com",
    "sessions": {
      "qwen2.5-vl-3b-awq/hf/ocr": "https://cloud.langfuse.com/project/.../sessions/..."
    }
  },
  "mlflow": {
    "tracking_uri": "file:./mlruns",
    "experiment_id": "12",
    "parent_run_id": "abc123",
    "per_model_per_runtime": {
      "tinyllama-w4a16/hf":   "12/run_uuid_a",
      "tinyllama-w4a16/vllm": "12/run_uuid_b"
    },
    "registered_models": ["triplequant/tinyllama-w4a16@2"]
  }
}
```

Appended to `results/_index.md` (top-level) — one entry per sweep, scannable.

### 12b.6 Failure-mode matrix

| Failure | W&B | Langfuse | MLflow | Local PNG/JSON |
|---|---|---|---|---|
| Missing creds | NoOp + warn | NoOp + warn | Still works (file:) | ✅ |
| Network blip mid-run | retry + warn; never crash | batched flush retries | local file ok | ✅ |
| Disk full | warn | warn | crashes mlflow `file:` write | ⚠️ — sweep aborts at next write |
| One tracker SDK ImportError | dropped from composite + warn | same | same | ✅ |
| All trackers fail | composite = NoOp | — | — | ✅ — run still produces full JSON/CSV/PNG |
| Subprocess OOM during eval | parent catches, logs error to all trackers, continues | same | same | partial JSON saved |

Rule: local outputs are the source of truth. Trackers are mirrors. Never block on a tracker.

### 12b.7 Quantize-time tracking (continuity link)

- [vLLM Quantization docs](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM ROCm install / MI300X](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)
- [llm-compressor repo](https://github.com/vllm-project/llm-compressor)
- [vLLM + AMD MI300X (AMD developer article)](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus.html)
- [jiwer (CER/WER)](https://github.com/jitsi/jiwer)
- [HuggingFace perplexity guide](https://huggingface.co/docs/transformers/perplexity)
- [MMLU dataset](https://huggingface.co/datasets/cais/mmlu)
- [linxy/LaTeX_OCR dataset](https://huggingface.co/datasets/linxy/LaTeX_OCR)
- [Weights & Biases Python SDK](https://docs.wandb.ai/ref/python/)
- [W&B Reports API (auto-generate)](https://docs.wandb.ai/guides/reports/create-a-report-programmatically)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
- [Langfuse Tracing & Scoring](https://langfuse.com/docs/tracing)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Transformers flavor](https://mlflow.org/docs/latest/llms/transformers/index.html)
