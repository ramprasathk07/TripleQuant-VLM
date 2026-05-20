"""
AWQ quantizer using llmcompressor.
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
        try:
            return self.model.config.num_hidden_layers
        except AttributeError:
            # Qwen2.5-VL nests language model config
            return self.model.config.language_config.num_hidden_layers
        
    
    