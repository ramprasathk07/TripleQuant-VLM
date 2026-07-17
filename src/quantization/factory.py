"""Quantizer factory for backend selection.

Routes QuantizeConfig to appropriate backend implementation (llm_compressor or modelopt).
"""
from __future__ import annotations

import logging
from src.config.schemas import QuantizeConfig
from .base import BaseQuantizer
from .llm_compressor import LLMCompressorQuantizer

logger = logging.getLogger(__name__)

def get_quantizer(config: QuantizeConfig) -> BaseQuantizer:
    """Instantiate quantizer for the configured backend.

    Args:
        config: QuantizeConfig specifying backend choice.

    Returns:
        BaseQuantizer subclass instance (LLMCompressorQuantizer or ModelOptQuantizer).

    Raises:
        ImportError: If modelopt backend is selected but nvidia-modelopt is not installed.
        ValueError: If backend is not recognized.
    """
    if config.backend == "llm_compressor":
        logger.info("Creating LLMCompressorQuantizer")
        return LLMCompressorQuantizer(config)
    elif config.backend == "modelopt":
        try:
            from .modelopt import ModelOptQuantizer
        except ImportError as e:
            raise ImportError(
                "nvidia-modelopt is required for backend='modelopt'.\n"
                "Install: pip install nvidia-modelopt[torch]\n"
                f"Original error: {e}"
            ) from e
        logger.info("Creating ModelOptQuantizer")
        return ModelOptQuantizer(config)
    elif config.backend == "torchao":
        try:
            from .torch_ao import TorchAOQuantizer
        except ImportError as e:
            raise ImportError(
                "torchao is required for backend='torchao'.\n"
                "Install: pip install torchao\n"
                f"Original error: {e}"
            ) from e
        logger.info("Creating TorchAOQuantizer")
        return TorchAOQuantizer(config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}. "
                         f"Choose 'llm_compressor', 'modelopt', or 'torchao'.")