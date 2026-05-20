#!/usr/bin/env python
"""
quantize.py — TripleQuant-VLM quantization entry point.

Usage:
    python quantize.py --config configs/quantize/qwen25vl_3b_awq.yaml
    python quantize.py --config configs/quantize/qwen25vl_3b_awq.yaml --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# ── Logging setup (before any project imports so all loggers inherit) ────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quantize")

# Config layer only — no torch / llmcompressor dependency at import time
from src.config import load_quantize_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TripleQuant-VLM — Quantize a vision-language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported methods: awq, gptq",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a quantize YAML config (e.g. configs/quantize/qwen25vl_3b_awq.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print the execution plan without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load + validate config (lightweight — no GPU) ────────────────────
    logger.info("Loading config: %s", args.config)
    try:
        config = load_quantize_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    method = config.quantization.method
    model_id = config.model.model_id
    suffix = config.output.save_dir_suffix
    save_name = model_id.split("/")[-1] + suffix
    output_dir = os.path.join(config.output.base_dir, save_name)

    logger.info("─" * 60)
    logger.info("Quantization plan:")
    logger.info("  Model        : %s", model_id)
    logger.info("  Method       : %s (W%dA16)", method.upper(), config.quantization.num_bits)
    logger.info("  Calib dataset: %s  [%d samples]",
                config.calibration.dataset_id, config.calibration.num_samples)
    logger.info("  Output dir   : %s", output_dir)
    logger.info("─" * 60)

    if args.dry_run:
        logger.info("--dry-run: stopping before model load. Config is valid ✓")
        return

    # ── 2. Import heavy ML stack (torch / llmcompressor) only when needed ────
    import src.quantization  # noqa: F401 — triggers @register decorators
    from src.quantization.registry import get_quantizer_class

    try:
        QuantizerClass = get_quantizer_class(method)
    except KeyError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    quantizer = QuantizerClass(config)

    # ── 3. Load model ────────────────────────────────────────────────────────
    quantizer.load_model()

    # ── 4. Build calibration dataset ─────────────────────────────────────────
    logger.info("Building calibration dataset from: %s", config.calibration.dataset_id)
    from src.data.processors import get_vlm_dataset, get_llm_dataset

    if quantizer.processor is not None:
        dataset = get_vlm_dataset(
            dataset_id=config.calibration.dataset_id,
            processor=quantizer.processor,
            split=config.calibration.dataset_split,
            num_samples=config.calibration.num_samples,
        )
    else:
        dataset = get_llm_dataset(
            dataset_id=config.calibration.dataset_id,
            tokenizer=quantizer.tokenizer,
            dataset_subset=config.calibration.dataset_subset,
            split=config.calibration.dataset_split,
            num_samples=config.calibration.num_samples,
        )

    # ── 5. Quantize + save ───────────────────────────────────────────────────
    os.makedirs(config.output.base_dir, exist_ok=True)
    quantizer.quantize(dataset, output_dir=output_dir)

    # Save the model and processor/tokenizer explicitly
    # Moving to CPU inside save() avoids accelerate/transformers offload KeyErrors
    quantizer.save(output_dir, weights=True)

    logger.info("✅ Quantization complete → %s", output_dir)


if __name__ == "__main__":
    main()
