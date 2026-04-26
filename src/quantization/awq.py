"""
AWQ quantizer using llmcompressor.

Registers itself as ``"awq"`` in the global quantizer registry.
"""
from __future__ import annotations

import logging

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.modifiers.quantization import QuantizationModifier

from src.config.schemas import QuantizeConfig
from .base import BaseQuantizer
from .registry import register

logger = logging.getLogger(__name__)


@register("awq")
class AWQQuantizer(BaseQuantizer):
    """AWQ (Activation-aware Weight Quantization) via llmcompressor."""

    def __init__(self, config: QuantizeConfig) -> None:
        super().__init__(config)
        # num_layers is inferred at quantize-time from the config model_id
        # so we don't hard-code 36 here — we detect from the model name.
        self._num_layers: int | None = None

    def _detect_num_layers(self) -> int:
        """Infer layer count from the loaded model's config."""
        try:
            return self.model.config.num_hidden_layers
        except AttributeError:
            # Qwen2.5-VL nests language model config
            return self.model.config.language_config.num_hidden_layers

    def _build_mappings(self, num_layers: int) -> list[AWQMapping]:
        """Build per-layer AWQ smooth/balance mappings for Qwen2.5 / Qwen2.5-VL."""
        is_vlm = "VL" in self.model_id or "vision" in self.model_id.lower()
        prefix = "model.language_model.layers" if is_vlm else "model.layers"

        mappings: list[AWQMapping] = []
        for i in range(num_layers):
            # Attention block
            mappings.append(
                AWQMapping(
                    smooth_layer=f"{prefix}.{i}.input_layernorm",
                    balance_layers=[
                        f"{prefix}.{i}.self_attn.q_proj",
                        f"{prefix}.{i}.self_attn.k_proj",
                        f"{prefix}.{i}.self_attn.v_proj",
                    ],
                )
            )
            # MLP block
            mappings.append(
                AWQMapping(
                    smooth_layer=f"{prefix}.{i}.post_attention_layernorm",
                    balance_layers=[
                        f"{prefix}.{i}.mlp.gate_proj",
                        f"{prefix}.{i}.mlp.up_proj",
                    ],
                )
            )
        return mappings

    def quantize(self, dataset, output_dir: str) -> None:
        """Run AWQ calibration + quantization, save result to *output_dir*."""
        q = self.config.quantization
        awq_params = q.awq  # guaranteed non-None by schema validator

        num_layers = self._detect_num_layers()
        logger.info("Detected %d transformer layers — building AWQ mappings …", num_layers)
        mappings = self._build_mappings(num_layers)

        recipe = [
            AWQModifier(
                duo_scaling=awq_params.duo_scaling,
                mappings=mappings,
            ),
            QuantizationModifier(
                ignore=q.ignore,
                config_groups={
                    "group_0": {
                        "targets": q.targets,
                        "weights": {
                            "num_bits": q.num_bits,
                            "type": "int",
                            "symmetric": q.symmetric,
                            "group_size": q.group_size,
                            "strategy": "group",
                            "dynamic": False,
                        },
                    }
                },
            ),
        ]

        proc_or_tok = self.processor if self.processor is not None else self.tokenizer
        cal = self.config.calibration

        logger.info(
            "Starting AWQ oneshot: %d calibration samples, max_seq=%d, batch=%d",
            cal.num_samples,
            cal.max_seq_length,
            cal.batch_size,
        )
        oneshot(
            model=self.model,
            processor=proc_or_tok,
            recipe=recipe,
            dataset=dataset,
            max_seq_length=cal.max_seq_length,
            num_calibration_samples=cal.num_samples,
            batch_size=cal.batch_size,
        )
        logger.info("AWQ quantization complete. (Weights will be saved to: %s)", output_dir)
