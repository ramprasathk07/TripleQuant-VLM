#!/usr/bin/env python
"""Main entry point for model quantization.

This module orchestrates a complete quantization workflow: loading a quantization
configuration from YAML, instantiating the appropriate backend (llm_compressor or
modelopt), loading the model from Hugging Face, and running quantization with
calibration data. The quantized model is saved to a descriptive output directory.

Usage:
    python quantize.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from src.config.schemas import QuantizeConfig
from src.quantization.factory import get_quantizer
from src.utils import setup_global_logging

# Configure logging to console and log/ directory
logger = setup_global_logging("quantize")



def main() -> None:
    """Load config, instantiate quantizer, and run quantization workflow.

    Parses command-line arguments for a YAML config file, validates the
    configuration, routes to the appropriate backend, loads the model, and
    invokes quantization with calibration data. Exits with code 1 on config
    file not found or validation failure.

    Raises:
        SystemExit: If config file not found or configuration is invalid.
    """
    parser = argparse.ArgumentParser(description="Quantize a model using a YAML configuration.")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to the quantization config YAML file."
    )
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    try:
        config = QuantizeConfig.model_validate(config_dict)
    except Exception as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    logger.info("Using backend: %s, method: %s", config.backend, config.method)

    quantizer = get_quantizer(config)

    logger.info("Loading model: %s", config.model.model_id)
    quantizer.load_model()

    logger.info("Starting quantization...")
    quantizer.quantize()

    logger.info("Quantization completed successfully. Output saved to: %s", config.output.output_dir)


if __name__ == "__main__":
    main()