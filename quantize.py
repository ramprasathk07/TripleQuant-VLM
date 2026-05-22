#!/usr/bin/env python
# quantize.py - Main entry point for model quantization
"""
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Quantize a model using a YAML configuration.")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to the quantization config YAML file."
    )
    args = parser.parse_args()

    # Load YAML config
    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Validate and create config object
    try:
        config = QuantizeConfig.model_validate(config_dict)
    except Exception as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    # Log the chosen backend and method
    logger.info("Using backend: %s, method: %s", config.backend, config.method)

    # Instantiate the appropriate quantizer (model not yet loaded)
    quantizer = get_quantizer(config)

    # Load model (and processor/tokenizer) from HuggingFace
    logger.info("Loading model: %s", config.model.model_id)
    quantizer.load_model()

    # Run quantization
    logger.info("Starting quantization...")
    quantizer.quantize()

    logger.info("Quantization completed successfully. Output saved to: %s", config.output.output_dir)


if __name__ == "__main__":
    main()