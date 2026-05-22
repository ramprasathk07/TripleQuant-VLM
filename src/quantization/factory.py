# src/quantization/factory.py
from __future__ import annotations

import logging
from src.config.schemas import QuantizeConfig
from .base import BaseQuantizer
from .llm_compressor import LLMCompressorQuantizer

logger = logging.getLogger(__name__)

def get_quantizer(config: QuantizeConfig) -> BaseQuantizer:
    """Factory to instantiate the appropriate quantizer based on config.backend."""
    if config.backend == "llm_compressor":
        logger.info("Creating LLMCompressorQuantizer")
        return LLMCompressorQuantizer(config)
    elif config.backend == "modelopt":
        try:
            from .modelOpy import ModelOptQuantizer
        except ImportError as e:
            raise ImportError(
                "nvidia-modelopt is required for backend='modelopt'.\n"
                "Install: pip install nvidia-modelopt[torch]\n"
                f"Original error: {e}"
            ) from e
        logger.info("Creating ModelOptQuantizer")
        return ModelOptQuantizer(config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}. "
                         f"Choose 'llm_compressor' or 'modelopt'.")