"""YAML configuration loaders for QuantizeConfig and BenchmarkConfig.

Parses YAML config files and validates against Pydantic schemas, raising
ValidationError with field-level details on constraint violations.
"""
from __future__ import annotations
from pathlib import Path

import yaml
from pydantic import ValidationError
from .schemas import BenchmarkConfig, QuantizeConfig

def _read_yaml(path: str | Path) -> dict:
    """Read and parse YAML file, raising FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_quantize_config(path: str | Path) -> QuantizeConfig:
    """Load and validate quantization config from YAML.

    Args:
        path: Path to YAML config file.

    Returns:
        QuantizeConfig: Validated quantization configuration.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If YAML does not conform to QuantizeConfig schema.
    """
    raw = _read_yaml(path)
    try:
        return QuantizeConfig.model_validate(raw)
    
    except ValidationError as exc:
        raise ValueError(
            f"Invalid quantize config '{path}':\n{exc}"
        ) from exc

def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate benchmark config from YAML.

    Args:
        path: Path to YAML config file.

    Returns:
        BenchmarkConfig: Validated benchmark configuration.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If YAML does not conform to BenchmarkConfig schema.
    """
    raw = _read_yaml(path)
    try:
        return BenchmarkConfig.model_validate(raw)
    
    except ValidationError as exc:
        raise ValueError(
            f"Invalid benchmark config '{path}':\n{exc}"
        ) from exc

