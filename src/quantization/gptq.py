"""
GPTQ quantizer using llmcompressor.

Registers itself as ``"gptq"`` in the global quantizer registry.
Optionally applies SmoothQuant before GPTQ.
"""
from __future__ import annotations

import logging

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

from src.config.schemas import QuantizeConfig
from .base import BaseQuantizer
from .registry import register

logger = logging.getLogger(__name__)


@register("gptq")
class GPTQQuantizer(BaseQuantizer):
    """GPTQ quantization via llmcompressor, with optional SmoothQuant pre-pass."""

    def quantize(self, dataset, output_dir: str) -> None:
        """Run GPTQ calibration + quantization, save result to *output_dir*."""
        q = self.config.quantization
        gptq_params = q.gptq  # guaranteed non-None by schema validator
        cal = self.config.calibration

        recipe = []

        if gptq_params.smoothquant_enabled:
            logger.info(
                "SmoothQuant enabled (strength=%.2f) — prepending SmoothQuantModifier",
                gptq_params.smoothquant_strength,
            )
            recipe.append(
                SmoothQuantModifier(
                    smoothing_strength=gptq_params.smoothquant_strength
                )
            )

        recipe.append(
            GPTQModifier(
                targets=q.targets,
                ignore=q.ignore,
                scheme=f"W{q.num_bits}A16",  # e.g. "W4A16" or "W8A16"
                block_size=gptq_params.block_size,
                dampening_frac=gptq_params.dampening_frac,
                sequential_update=gptq_params.sequential_update,
            )
        )

        proc_or_tok = self.processor if self.processor is not None else self.tokenizer

        logger.info(
            "Starting GPTQ oneshot: %d calibration samples, max_seq=%d, batch=%d",
            cal.num_samples,
            cal.max_seq_length,
            cal.batch_size,
        )
        oneshot(
            model=self.model,
            dataset=dataset,
            processor=proc_or_tok,
            recipe=recipe,
            max_seq_length=cal.max_seq_length,
            num_calibration_samples=cal.num_samples,
            batch_size=cal.batch_size,
        )
        logger.info("GPTQ quantization complete. (Weights will be saved to: %s)", output_dir)
