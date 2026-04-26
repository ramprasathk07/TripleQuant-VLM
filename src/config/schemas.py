"""
Pydantic v2 schemas for all TripleQuant-VLM config files.

Two root models:
  - QuantizeConfig  → configs/quantize/*.yaml
  - BenchmarkConfig → configs/benchmark/*.yaml
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────
# Shared
# ─────────────────────────────────────────────────────────────

class CalibrationConfig(BaseModel):
    dataset_id: str
    dataset_subset: Optional[str] = None
    dataset_split: str = "train"
    num_samples: int = 64
    max_seq_length: int = 512
    batch_size: int = 1


class OutputConfig(BaseModel):
    base_dir: str = "outputs"
    save_compressed: bool = True
    save_dir_suffix: str = "-quantized"


# ─────────────────────────────────────────────────────────────
# Quantize workflow
# ─────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    model_id: str
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None


class AWQParams(BaseModel):
    duo_scaling: bool = False


class GPTQParams(BaseModel):
    block_size: int = 128
    dampening_frac: float = 0.01
    sequential_update: bool = True
    # SmoothQuant integration
    smoothquant_enabled: bool = False
    smoothquant_strength: float = 0.5


class QuantMethodConfig(BaseModel):
    method: Literal["awq", "gptq"]
    num_bits: int = 4
    group_size: int = 128
    symmetric: bool = True
    targets: list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=list)
    # Method-specific sub-configs (optional; provide only the relevant one)
    awq: Optional[AWQParams] = None
    gptq: Optional[GPTQParams] = None

    @model_validator(mode="after")
    def _inject_defaults(self) -> QuantMethodConfig:
        """Ensure the matching sub-config always has a value."""
        if self.method == "awq" and self.awq is None:
            self.awq = AWQParams()
        if self.method == "gptq" and self.gptq is None:
            self.gptq = GPTQParams()
        return self


class QuantizeConfig(BaseModel):
    """Root schema for configs/quantize/*.yaml"""
    model: ModelConfig
    quantization: QuantMethodConfig
    calibration: CalibrationConfig
    output: OutputConfig


# ─────────────────────────────────────────────────────────────
# Benchmark workflow
# ─────────────────────────────────────────────────────────────

class BenchmarkModelEntry(BaseModel):
    name: str
    path: str
    is_local: bool = False
    # Optional per-model overrides
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096


class DatasetConfig(BaseModel):
    dataset_id: str
    dataset_subset: Optional[str] = None
    dataset_split: str = "test"
    num_samples: int = 50
    prompts: list[str] = Field(default_factory=lambda: ["Transcribe the text in this image."])


class MetricsConfig(BaseModel):
    latency: bool = True
    accuracy: bool = True
    memory: bool = True
    perplexity: bool = False
    warmup_runs: int = 2
    timed_runs: int = 5
    max_new_tokens: int = 512


class BenchmarkConfig(BaseModel):
    """Root schema for configs/benchmark/*.yaml"""
    models: list[BenchmarkModelEntry]
    dataset: DatasetConfig
    metrics: MetricsConfig
    results_dir: str = "results"
