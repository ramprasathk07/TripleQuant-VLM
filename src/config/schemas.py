"""Pydantic v2 schemas for TripleQuant-VLM quantization and benchmarking.

This module defines the single source of truth for all configuration: quantization
schemes, model loading, calibration, output paths, and benchmarking. Each schema
is validated at instantiation and includes custom validators for constraints
(e.g., group_size must be a power of 2, block_size must be 2D positive integers).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Enums / Literals
class QuantScheme(str, Enum):
    """All supported quantization bit-schemes and data types."""
    W4A16 = "W4A16"
    W4A16_ASYM = "W4A16_ASYM"
    W4A8 = "W4A8"
    W8A8 = "W8A8"
    W8A8_ASYM = "W8A8_ASYM"
    W8A16 = "W8A16"
    FP8 = "FP8"
    FP8_DYNAMIC = "FP8_DYNAMIC"
    FP8_BLOCK = "FP8_BLOCK"
    NVFP4 = "NVFP4"
    NVFP4A16 = "NVFP4A16"
    MXFP4 = "MXFP4"


QuantSchemeLiteral = Literal[
    "W4A16", "W4A16_ASYM", "W4A8",
    "W8A8", "W8A8_ASYM", "W8A16",
    "FP8", "FP8_DYNAMIC", "FP8_BLOCK",
    "NVFP4", "NVFP4A16", "MXFP4",
]

MethodLiteral = Literal["awq", "gptq", "ptq", "turboquant"]
ModalityLiteral = Literal["auto", "text", "vision", "vision_text", "audio"]
ModelTypeLiteral = Literal["llm", "vlm", "moe", "custom"]
# Which transformers Auto* class to load a benchmark model with. "auto" picks
# from model_type (vlm -> image_text_to_text, else causal_lm); the explicit values
# force a class for architectures the heuristic misses.
ModelClassLiteral = Literal[
    "auto", "causal_lm", "image_text_to_text", "seq2seq_lm", "vision2seq",
]

# New type for observers
ObserverLiteral = Literal["mse", "minmax", "maxabs", "percentile"]

# Sub-configs
class ModelConfig(BaseModel):
    """Hugging Face model loading configuration.

    Controls model instantiation, dtype, device placement, and modality hints.
    Used during quantization (config.model) and benchmarking (BenchmarkModelEntry).

    Attributes:
        model_id: Hugging Face model identifier (e.g., 'meta-llama/Llama-2-7b').
        torch_dtype: PyTorch dtype string (e.g., 'bfloat16', 'float16').
        device_map: Device placement strategy for multi-GPU ('auto', 'cuda', etc.).
        trust_remote_code: Allow loading custom code from model's repo (required for some VLMs).
        model_type: Classification hint for routing ('llm', 'vlm', 'moe', 'custom').
        modality: Modality hint ('auto', 'text', 'vision', 'vision_text', 'audio').
        min_pixels: Minimum image pixels for VLM processors (dynamic resolution).
        max_pixels: Maximum image pixels for VLM processors (dynamic resolution).
    """
    model_id: str
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    model_type: ModelTypeLiteral = "llm"
    modality: ModalityLiteral = "auto"
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None


class SchemeConfig(BaseModel):
    """Quantization scheme and bit-width configuration.

    Controls precision (W4A16, W8A8, FP8, etc.), group size for grouped quantization,
    symmetry, activation order heuristic, and observer/calibration strategy.

    Attributes:
        scheme: Quantization bit-width scheme (e.g., 'W4A16', 'W8A8', 'FP8_DYNAMIC').
        group_size: Granularity for grouped quantization. Must be power of 2 (e.g., 128).
        symmetric: If True, use symmetric quantization; else asymmetric (with zero-point).
        actorder: If True, use activation-order heuristic (AWQ/GPTQ optimization).
        block_size: Block dimensions for FP8_BLOCK scheme, [M, N] where both > 0. None for other schemes.
        targets: Layer module types to quantize (e.g., ['Linear']; empty = all).
        ignore: Layer name patterns to exclude from quantization (regex or literal).
        observer: Calibration statistics observer: 'mse', 'minmax', 'maxabs', 'percentile'.
        per_channel: If True, compute separate quantization ranges per output channel.
        dynamic_activations: If True, quantize activations dynamically per sample.
    """
    model_config = ConfigDict(extra="forbid")

    scheme: QuantSchemeLiteral = "W4A16"
    group_size: int = 128
    symmetric: bool = True
    actorder: bool = False
    block_size: Optional[list[int]] = Field(
        default=None,
        description="Block size for FP8_BLOCK scheme.",
    )
    targets: list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=lambda: ["lm_head"])

    observer: ObserverLiteral = "mse"
    per_channel: bool = False
    dynamic_activations: bool = False

    @field_validator("group_size")
    @classmethod
    def _gs_positive(cls, v: int) -> int:
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError(f"group_size must be a positive power of 2, got {v}")
        return v

    @field_validator("block_size")
    @classmethod
    def _block_size_2d(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is None:
            return v
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError(f"block_size must be 2 positive ints [M, N], got {v}")
        return v


class CalibrationConfig(BaseModel):
    """Calibration dataset and preprocessing configuration for quantization.

    Specifies the dataset source, number of samples, sequence length, field names
    (for text and image columns), and format detection strategy (auto, chat, or image_text).

    Attributes:
        dataset_name: Hugging Face dataset identifier (e.g., 'HuggingFaceH4/ultrachat_200k').
        subset: HF dataset config name for multi-config datasets (e.g., 'full' for linxy/LaTeX_OCR).
                Required when dataset has multiple configs; None for single-config datasets.
        num_samples: Number of calibration samples to use (for computing quantization statistics).
        max_seq_len: Maximum sequence length for tokenized inputs (pads/truncates to this).
        split: Dataset split to load ('train', 'validation', 'test').
        image_field: Column name for image data in VLM datasets (None = auto-detect).
        text_field: Column name for text data (None = auto-detect).
        seed: Random seed for reproducible sample selection.
        dataset_format: Auto-detect or hint: 'auto', 'chat', 'image_text'.
        instruction_prompt: Custom instruction prompt for dataset-specific formatting.
                           None = auto-select based on dataset name.
    """
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    subset: Optional[str] = None
    num_samples: int = 512
    max_seq_len: int = 2048
    split: str = "train"
    image_field: Optional[str] = "image"
    text_field: Optional[str] = "text"
    seed: int = 42
    dataset_format:     str          = "auto"
    instruction_prompt: Optional[str] = None

    @field_validator("num_samples", "max_seq_len")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class OutputConfig(BaseModel):
    """Output and model export configuration.

    Specifies output directory, whether to save in compressed format, processor/tokenizer
    preservation, and optional Hugging Face Hub integration.

    Attributes:
        output_dir: Directory where quantized model is saved (expands ~ to home).
        save_compressed: If True, use safetensors compressed format; else standard safetensors.
        save_processor: If True, save VLM processor or LLM tokenizer alongside model.
        push_to_hub: Optional Hugging Face Hub repo ID to push quantized model to.
    """
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("./output")
    save_compressed: bool = True
    save_processor: bool = True
    push_to_hub: Optional[str] = None

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v) -> Path:
        return Path(v).expanduser()


class SmoothQuantConfig(BaseModel):
    """SmoothQuant activation quantization configuration.

    Enables layer-wise activation quantization with per-layer strength and module
    mappings that define which quantization layers follow which linear layers.

    Attributes:
        enabled: Whether SmoothQuant is applied (default True when this config present).
        strength: Per-layer smooth strength factor (0.0-1.0; higher = more aggressive).
        mappings: List of [quantization_layers, associated_linear_layer] pairs,
                  where each layer can be a regex pattern ('re:...') or literal name.
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    strength: float = 0.5
    mappings: List[List[Union[List[str], str]]] = [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    ]


class AWQParams(BaseModel):
    """AWQ-specific hyperparameters.

    Attributes:
        duo_scaling: If True, use duo-scaling variant for improved accuracy.
    """
    model_config = ConfigDict(extra="forbid")

    duo_scaling: bool = False


class GPTQParams(BaseModel):
    """GPTQ-specific hyperparameters.

    Attributes:
        dampening_frac: Damping factor for Hessian computation (0.01 typical).
        block_size: Hessian block size for gradient computation (128 typical).
        sequential_update: If True, update weights sequentially per block.
    """
    model_config = ConfigDict(extra="forbid")

    dampening_frac: float = 0.01
    block_size: int = 128
    sequential_update: bool = True


# Top-level QuantizeConfig
# Scheme/method compatibility sets (llm_compressor backend).
_AWQ_ALLOWED_SCHEMES: set[str] = {"W4A16", "W4A16_ASYM"}
_GPTQ_ALLOWED_SCHEMES: set[str] = {
    "W4A16", "W4A16_ASYM", "W4A8", "W8A16", "W8A8", "W8A8_ASYM",
}
# Float-typed schemes have no algorithm variant — PTQ only.
_FLOAT_SCHEMES: set[str] = {
    "FP8", "FP8_DYNAMIC", "FP8_BLOCK", "NVFP4", "NVFP4A16", "MXFP4",
}
# Schemes llm_compressor cannot build — route to backend=modelopt.
_MODELOPT_ONLY_SCHEMES: set[str] = {"FP8_BLOCK", "MXFP4"}

BackendLiteral = Literal["llm_compressor", "modelopt", "torchao"]


class QuantizeConfig(BaseModel):
    """Top-level quantization configuration combining method, scheme, backend, and tuning options.

    Orchestrates model loading, calibration data, quantization scheme/backend selection,
    and optional tuning (AWQ duo-scaling, GPTQ dampening, SmoothQuant). Validates
    method-scheme compatibility (e.g., AWQ only supports W4A16 variants).

    Attributes:
        method: Quantization algorithm ('awq', 'gptq', 'ptq', 'turboquant').
        backend: Quantization framework ('llm_compressor' or 'modelopt').
        model: Model loading configuration (model_id, dtype, device_map, etc.).
        scheme: Quantization scheme configuration (bit-width, group_size, observer, etc.).
        calibration: Calibration dataset configuration (dataset_name, num_samples, etc.).
        output: Output export configuration (output_dir, compression, Hub push).
        smoothquant: Optional SmoothQuant activation quantization config.
        awq: Optional AWQ-specific hyperparameters.
        gptq: Optional GPTQ-specific hyperparameters.
    """
    method: MethodLiteral
    backend: BackendLiteral = "llm_compressor"
    model: ModelConfig
    scheme: SchemeConfig = Field(default_factory=SchemeConfig)
    calibration: CalibrationConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    smoothquant: Optional[SmoothQuantConfig] = None
    awq: Optional[AWQParams] = None
    gptq: Optional[GPTQParams] = None

    @model_validator(mode="after")
    def _validate_method_scheme_compat(self) -> QuantizeConfig:
        # Only the llm_compressor backend uses these recipe constraints.
        # modelopt and turboquant have their own scheme handling.
        if self.backend != "llm_compressor" or self.method == "turboquant":
            return self

        scheme = self.scheme.scheme

        if scheme in _MODELOPT_ONLY_SCHEMES:
            raise ValueError(
                f"scheme='{scheme}' is not supported by backend='llm_compressor'. "
                f"Set backend='modelopt' for {sorted(_MODELOPT_ONLY_SCHEMES)}."
            )

        if self.method == "awq" and scheme not in _AWQ_ALLOWED_SCHEMES:
            raise ValueError(
                f"method='awq' requires scheme in {sorted(_AWQ_ALLOWED_SCHEMES)}, "
                f"got '{scheme}'."
            )

        if self.method == "gptq" and scheme not in _GPTQ_ALLOWED_SCHEMES:
            raise ValueError(
                f"method='gptq' requires an INT scheme in {sorted(_GPTQ_ALLOWED_SCHEMES)}, "
                f"got '{scheme}'. Float schemes (FP8/NVFP4) need method='ptq'."
            )

        if scheme in _FLOAT_SCHEMES and self.method != "ptq":
            raise ValueError(
                f"scheme='{scheme}' is float-typed and has no algorithm variant — "
                f"use method='ptq' (got method='{self.method}')."
            )

        return self

    @model_validator(mode="after")
    def _inject_method_defaults(self) -> QuantizeConfig:
        if self.method == "awq" and self.awq is None:
            self.awq = AWQParams()
        if self.method == "gptq" and self.gptq is None:
            self.gptq = GPTQParams()
        return self


class MetricsConfig(BaseModel):
    """Benchmark metrics selection for evaluation runs.

    Attributes:
        quality_llm: Text quality metrics ('ppl'=perplexity, 'mmlu_tiny', 'gsm8k', 'aime',
                    'ceval', 'humaneval', 'logit_kl', 'token_agree').
        quality_ocr: VLM OCR metrics ('cer'=character error rate, 'wer', 'exact_match', 'bleu').
        perf: Performance metrics ('ttft'=time-to-first-token, 'tpot'=time-per-output-token,
              'throughput', 'max_context', 'ctx_sweep').
        memory: Memory metrics ('disk'=model size, 'vram'=peak GPU, 'load_time').
    """
    quality_llm: list[Literal[
        "ppl", "mmlu_tiny", "logit_kl", "token_agree",
        "gsm8k", "ceval", "humaneval", "aime",
    ]] = ["ppl"]
    quality_ocr: list[Literal["cer", "wer", "exact_match", "bleu"]] = ["cer"]
    perf:        list[Literal["ttft", "tpot", "throughput", "ctx_sweep", "max_context", "tq_bits_sweep"]] = ["throughput", "ttft", "tpot"]
    memory:      list[Literal["disk", "vram", "load_time"]] = ["disk", "vram"]

class EvalDatasetConfig(BaseModel):
    """Benchmark dataset and metric configuration for quality evaluations.

    Groups dataset names, splits, sample counts, and hyperparameters for each metric
    (perplexity, MMLU, GSM8K, AIME, C-Eval, HumanEval, OCR). Centralizes all evaluation
    data sources in one place for reproducibility.

    Attributes:
        ppl_dataset, ppl_subset: Perplexity evaluation dataset.
        mmlu_subjects: MMLU subjects to evaluate (subset of cais/mmlu).
        mmlu_num_q_per_subject: Questions sampled per subject; total n = subjects * this.
        gsm8k_dataset, gsm8k_num_samples, gsm8k_max_new_tokens: Grade-school math config.
        aime_dataset, aime_split, aime_num_samples, aime_max_new_tokens: Competition math config.
        ceval_dataset, ceval_subjects, ceval_num_q_per_subject: Chinese MMLU config.
        humaneval_dataset, humaneval_num_samples, humaneval_max_new_tokens: Code gen config.
        humaneval_execute: If True, execute model-generated code (SECURITY: only in sandbox).
        humaneval_timeout: Timeout in seconds for code execution.
        ocr_dataset, ocr_datasets, ocr_subset, ocr_image_field, ocr_text_field: OCR config.
        ocr_split, ocr_num_samples, ocr_max_new_tokens: OCR evaluation parameters.
    """
    # ── Perplexity ──
    ppl_dataset: str = "wikitext"
    ppl_subset: str = "wikitext-2-raw-v1"

    # ── MMLU (valid cais/mmlu subject configs only) ──
    mmlu_subjects: list[str] = ["high_school_mathematics", "high_school_computer_science",
                                "philosophy", "high_school_world_history", "global_facts"]
    # Questions per subject. Total n = len(mmlu_subjects) * this. Drives the noise
    # floor: n=250 gives a binomial SE of ~3.1pp (95% CI ~+/-6pp), so sub-2pp
    # accuracy gaps between two models are indistinguishable from sampling noise.
    # Raise it (and/or add subjects) when a comparison needs to resolve small deltas.
    mmlu_num_q_per_subject: int = 50

    # ── GSM8K (grade-school math, generative) ──
    gsm8k_dataset: str = "openai/gsm8k"
    gsm8k_num_samples: int = 200
    gsm8k_max_new_tokens: int = 512

    # ── AIME (competition math, generative) ──
    aime_dataset: str = "Maxwell-Jia/AIME_2024"
    aime_split: str = "train"
    aime_num_samples: int = 30
    aime_max_new_tokens: int = 1024

    # ── C-Eval (Chinese MC, logit-scored) ──
    ceval_dataset: str = "ceval/ceval-exam"
    ceval_subjects: list[str] = ["computer_network", "high_school_mathematics",
                                 "logic", "college_programming", "marxism"]
    ceval_num_q_per_subject: int = 20

    # ── HumanEval (code, pass@1) ──
    humaneval_dataset: str = "openai/openai_humaneval"
    humaneval_num_samples: int = 20
    humaneval_max_new_tokens: int = 512
    humaneval_execute: bool = False          # SECURITY: runs model-generated code if True
    humaneval_timeout: float = 10.0

    # ── OCR (VLM) ──
    ocr_dataset: str = "linxy/LaTeX_OCR"
    ocr_datasets: Optional[list[str]] = None  # if set, eval each; else just ocr_dataset
    ocr_subset: Optional[str] = None          # HF config name; None → registry default (e.g. linxy→"full")
    ocr_image_field: Optional[str] = None     # None → auto-detect
    ocr_text_field: Optional[str] = None      # None → auto-detect
    ocr_split: str = "test"
    ocr_num_samples: int = 500
    ocr_max_new_tokens: int = 256

class LatencyConfig(BaseModel):
    """Performance profiling parameters for latency and throughput benchmarks.

    Attributes:
        prompt_lens: Input prompt lengths to test (used for context-sweep benchmarks).
        output_lens: Output token counts to generate (first value used for throughput).
        batch_sizes: Batch sizes to sweep for throughput measurement.
        ctx_sweep: Context lengths to binary-search for max-context capacity.
        num_requests: Number of timed requests per latency benchmark trial.
        warmup_requests: Warm-up iterations before timing (excluded from statistics).
    """
    prompt_lens: list[int] = [512]
    output_lens: list[int] = [128]
    batch_sizes: list[int] = [1, 4, 8, 16]
    ctx_sweep:   list[int] = [512, 2048, 8192]
    num_requests: int = 100
    warmup_requests: int = 5

    # TurboQuant bits x accuracy x context-length sweep (perf metric "tq_bits_sweep").
    # Each [key_bits, value_bits] pair is characterized at each context length; accuracy
    # = next-token agreement vs FP16 KV. Lengths must exceed tq_sweep_ring to engage the
    # compressed store.
    tq_bits_pairs:     list[list[int]] = [[2, 2], [3, 2], [4, 4], [8, 8]]
    tq_sweep_lengths:  list[int]       = [512, 1024, 2048, 4096]
    tq_sweep_ring:     int             = 256

class TurboQuantRuntimeConfig(BaseModel):
    """Per-model TurboQuant KV-cache settings for benchmarking.

    When ``enabled``, the HF runtime patches attention to read from a TurboQuant
    cache (ring buffer of recent exact tokens + bit-compressed overflow store), so
    generation runs over the compressed KV. Single-sequence only (the cache assumes
    batch size 1); batched throughput falls back to the normal cache.

    Attributes:
        enabled: Turn TurboQuant on for this model entry.
        ring_capacity: Most-recent tokens kept exact (bf16) before compression.
        key_bits / value_bits: Bit budget for compressed keys / values.
        use_qjl: Keep the QJL residual on key reconstruction (default False — it
                 adds variance and hurts generation; MSE-only is better).
        use_compressed_store: Read the compressed overflow in attention (True = TQ
                 in the decode loop) vs ring-only sliding window (False).
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    ring_capacity: int = 256
    key_bits: int = 3
    value_bits: int = 2
    use_qjl: bool = False
    use_compressed_store: bool = True


class BenchmarkModelEntry(BaseModel):
    """Single model entry in a benchmark comparison run.

    Attributes:
        name: Display name for the model (e.g., 'TinyLlama 1.1B').
        path: Local directory or Hugging Face model ID.
        is_local: If True, load from local disk; else from Hugging Face Hub.
        is_compressed: If True, use safetensors compressed format.
        model_type: Model classification ('llm', 'vlm', 'moe', 'custom').
        backend_hint: Optional quantization backend hint for runtime (overrides auto-detect).
        vllm_quantization: vLLM quantization format ('compressed-tensors', 'modelopt', etc.).
        gpu_memory_utilization: vLLM GPU utilization target (0.0-1.0, typical 0.85).
        max_model_len: Maximum sequence length for generation (vLLM KV-cache limit).
        skip_on: Skip this model on GPU arch list (e.g., ['sm_86'] for RTX 3060).
        dtype: Inference dtype ('auto', 'bfloat16', 'float16', 'float32').
        device_map: Device placement strategy ('auto', 'cuda', etc.).
        trust_remote_code: Allow loading custom code from model's repo.
        tensor_parallel_size: vLLM tensor parallelism degree (> 1 for multi-GPU).
        hf_quantization: Hugging Face on-the-fly BitsAndBytes quantization ('int8', 'int4', 'nf4').
    """
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

    # ── Runtime loading knobs (consumed directly by HFRuntime / VLLMRuntime) ──
    dtype: str = "auto"                       # 'auto' | 'bfloat16' | 'float16' | 'float32'
    device_map: str = "auto"                  # HF device placement
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1             # vLLM multi-GPU
    hf_quantization: Optional[str] = None     # HF on-the-fly BnB: None | 'int8' | 'int4' | 'nf4'
    torchao: Optional[Literal["int4wo", "int8wo", "int8dq", "fp8wo"]] = None  # torchao on-the-fly quant (HF only)
    model_class: ModelClassLiteral = "auto"   # which transformers Auto* class to load with

    # ── TurboQuant KV-cache (HF runtime only) ──
    turboquant: Optional[TurboQuantRuntimeConfig] = None

    @property
    def is_vlm(self) -> bool:
        """True if this entry describes a vision-language model."""
        return self.model_type == "vlm" or self.model_class in ("image_text_to_text", "vision2seq")

    @property
    def turboquant_enabled(self) -> bool:
        """True if TurboQuant is configured and enabled for this entry."""
        return self.turboquant is not None and self.turboquant.enabled

class TrackingConfig(BaseModel):
    """Multi-backend experiment tracking configuration.

    Enables logging to Weights & Biases, LangFuse, and/or MLflow. Local PNG plots
    are always generated; experiment trackers are additive (can enable multiple).

    Attributes:
        enabled: List of tracking backends to use (union of 'wandb', 'langfuse', 'mlflow').
        wandb_project: Weights & Biases project name.
        wandb_entity: W&B entity/team name (None = personal account).
        wandb_tags: List of tags for W&B experiment filtering.
        wandb_public: If True, create a public shareable W&B project link.
        wandb_api_key_env: Environment variable name for W&B API key.
    """
    enabled: list[Literal["wandb", "langfuse", "mlflow", "tensorboard"]] = ["wandb", "tensorboard"]

    wandb_project: str = "triplequant-vlm"
    wandb_entity: Optional[str] = None
    wandb_tags: list[str] = []
    wandb_public: bool = True
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
    runtimes: list[Literal["hf", "vllm"]] = ["hf"]   # run each model on these runtimes
    metrics: MetricsConfig = MetricsConfig()
    datasets: EvalDatasetConfig = EvalDatasetConfig()
    latency: LatencyConfig = LatencyConfig()
    tracking: TrackingConfig = TrackingConfig()
    crash_safe: bool = True
    seed: int = 42
    hf_token_env: str = "HF_TOKEN"
