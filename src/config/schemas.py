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


class BenchmarkModelEntry(BaseModel):
    """Single model entry for benchmarking with vLLM serving configuration."""
    name: str
    path: str
    is_local: bool = False
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096


class TrackingConfig(BaseModel):
    """Experiment tracking and result logging configuration for benchmarks.

    Supports Weights and Biases (wandb), Langfuse, and MLflow integrations
    with local fallback to filesystem output.
    """
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    langfuse_enabled: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "TripleQuant-VLM"
    local_output_dir: Path = Path("./benchmark_results")

    @field_validator("local_output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v) -> Path:
        return Path(v).expanduser()


class BenchmarkConfig(BaseModel):
    """Benchmark task configuration for vLLM serving evaluation.

    Specifies models to benchmark, evaluation tasks (text, OCR), calibration dataset,
    and tracking/logging destinations for results.
    """
    models: List[BenchmarkModelEntry] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=lambda: ["text", "ocr"])
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    ocr_dataset_name: str = "linxy/LaTeX_OCR"
    num_samples: int = 100
    max_new_tokens: int = 256
    seed: int = 42
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)