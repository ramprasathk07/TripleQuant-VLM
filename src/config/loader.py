"""
Config loader: YAML → validated Pydantic model.

Usage:
    from src.config import load_quantize_config, load_benchmark_config

    q_cfg = load_quantize_config("configs/quantize/qwen25vl_3b_awq.yaml")
    b_cfg = load_benchmark_config("configs/benchmark/ocr_comparison.yaml")
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schemas import BenchmarkConfig, QuantizeConfig


def _read_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_quantize_config(path: str | Path) -> QuantizeConfig:
    """Load and validate a quantize config YAML.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: if the YAML fails schema validation.
    """
    raw = _read_yaml(path)
    try:
        return QuantizeConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid quantize config '{path}':\n{exc}"
        ) from exc


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark config YAML.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: if the YAML fails schema validation.
    """
    raw = _read_yaml(path)
    try:
        return BenchmarkConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid benchmark config '{path}':\n{exc}"
        ) from exc
