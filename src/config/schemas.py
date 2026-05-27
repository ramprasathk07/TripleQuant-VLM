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
    W8A8 = "W8A8"
    W8A8_ASYM = "W8A8_ASYM"
    W8A16 = "W8A16"
    FP8 = "FP8"
    FP8_DYNAMIC = "FP8_DYNAMIC"
    FP8_BLOCK = "FP8_BLOCK"
    NVFP4 = "NVFP4"
    MXFP4 = "MXFP4"


QuantSchemeLiteral = Literal[
    "W4A16", "W4A16_ASYM",
    "W8A8", "W8A8_ASYM", "W8A16",
    "FP8", "FP8_DYNAMIC", "FP8_BLOCK",
    "NVFP4", "MXFP4",
]

MethodLiteral = Literal["awq", "gptq", "turboquant"]
ModalityLiteral = Literal["auto", "text", "vision", "audio"]
ModelTypeLiteral = Literal["llm", "vlm", "moe", "custom"]

# New type for observers
ObserverLiteral = Literal["mse", "minmax", "maxabs", "percentile"]

# Sub-configs
class ModelConfig(BaseModel):
    """Hugging Face model loading configuration including dtype, device placement, and vision/text modality hints."""
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
    """
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
    """
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    num_samples: int = 512
    max_seq_len: int = 2048
    split: str = "train"
    image_field: Optional[str] = "image"
    text_field: Optional[str] = "text"
    seed: int = 42
    dataset_format:     str          = "auto"   # "auto" | "chat" | "image_text"
    instruction_prompt: Optional[str] = None    # None = auto-select based on dataset
    
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

    Enables layer-wise smooth quantization with per-layer strength and module
    mappings that define which quantization layers follow which linear layers.
    """
    enabled: bool = True
    strength: float = 0.5
    mappings: List[List[Union[List[str], str]]] = [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    ]


class AWQParams(BaseModel):
    """AWQ-specific hyperparameters."""
    duo_scaling: bool = False


class GPTQParams(BaseModel):
    """GPTQ-specific hyperparameters."""
    dampening_frac: float = 0.01
    sequential_update: bool = True


# Top-level QuantizeConfig
_AWQ_ALLOWED_SCHEMES: set[str] = {"W4A16", "W4A16_ASYM"}
BackendLiteral = Literal["llm_compressor", "modelopt"]


class QuantizeConfig(BaseModel):
    """Top-level quantization configuration combining method, scheme, backend, and tuning options.

    Orchestrates model loading, calibration data, quantization scheme/backend selection,
    and optional tuning (AWQ duo-scaling, GPTQ dampening, SmoothQuant). Validates
    method-scheme compatibility (e.g., AWQ only supports W4A16 variants).
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
        scheme = self.scheme.scheme
        if self.method == "awq" and scheme not in _AWQ_ALLOWED_SCHEMES:
            raise ValueError(
                f"method='awq' requires scheme in {_AWQ_ALLOWED_SCHEMES}, "
                f"got '{scheme}'."
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

    # ── Runtime loading knobs (consumed directly by HFRuntime / VLLMRuntime) ──
    dtype: str = "auto"                       # 'auto' | 'bfloat16' | 'float16' | 'float32'
    device_map: str = "auto"                  # HF device placement
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1             # vLLM multi-GPU
    hf_quantization: Optional[str] = None     # HF on-the-fly BnB: None | 'int8' | 'int4' | 'nf4'

    @property
    def is_vlm(self) -> bool:
        """True if this entry describes a vision-language model."""
        return self.model_type == "vlm"

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
    runtimes: list[Literal["hf", "vllm"]] = ["hf"]   # run each model on these runtimes
    metrics: MetricsConfig = MetricsConfig()
    datasets: EvalDatasetConfig = EvalDatasetConfig()
    latency: LatencyConfig = LatencyConfig()
    tracking: TrackingConfig = TrackingConfig()
    crash_safe: bool = True
    seed: int = 42
    hf_token_env: str = "HF_TOKEN"
