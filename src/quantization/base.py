"""
Abstract base class for all quantizers.

Sub-classes must implement ``quantize(dataset, output_dir)``.
All logging goes through the Python ``logging`` module — no raw ``print`` calls.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from src.config.schemas import QuantizeConfig

logger = logging.getLogger(__name__)


class BaseQuantizer(ABC):
    """Base class for all quantization strategies.

    Args:
        config: A validated :class:`~src.config.schemas.QuantizeConfig`.
    """

    def __init__(self, config: QuantizeConfig) -> None:
        self.config = config
        self.model_id: str = config.model.model_id
        self.torch_dtype = getattr(torch, config.model.torch_dtype)
        self.device_map: str = config.model.device_map
        self.trust_remote_code: bool = config.model.trust_remote_code

        self.model = None
        self.processor = None
        self.tokenizer = None

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """Load model and processor/tokenizer from HuggingFace hub or local path."""
        logger.info("Loading model: %s (dtype=%s, device_map=%s)",
                    self.model_id, self.config.model.torch_dtype, self.device_map)

        is_vlm = "VL" in self.model_id or "vision" in self.model_id.lower()

        if is_vlm:
            self._load_vlm()
        else:
            self._load_llm()

    def _load_vlm(self) -> None:
        processor_kwargs: dict = {}
        if self.config.model.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.model.min_pixels
        if self.config.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.model.max_pixels

        logger.info("Detected VLM — loading AutoProcessor + AutoModelForImageTextToText")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            **processor_kwargs,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )

    def _load_llm(self) -> None:
        logger.info("Detected LLM — loading AutoTokenizer + AutoModelForCausalLM")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def quantize(self, dataset, output_dir: str) -> None:
        """Run quantization on *dataset* and save weights to *output_dir*."""

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save(self, output_dir: str, weights: bool = True) -> None:
        """Save model weights and/or processor/tokenizer to *output_dir*.

        Args:
            output_dir: Target directory.
            weights: If ``True``, also save model weights (move to CPU first
                to avoid the multi-modal ``module_map`` KeyError).
        """
        if weights:
            logger.info("Moving model to CPU before saving …")
            self.model.to("cpu")
            # --- FIX FOR TRANSFORMERS VLM MODULE_MAP BUG ---
            if hasattr(self.model, "hf_device_map"):
                del self.model.hf_device_map
            # -----------------------------------------------
            logger.info("Saving quantized model weights to: %s", output_dir)
            self.model.save_pretrained(
                output_dir,
                save_compressed=self.config.output.save_compressed,
            )

        if self.processor is not None:
            logger.info("Saving processor to: %s", output_dir)
            self.processor.save_pretrained(output_dir)

        if self.tokenizer is not None:
            logger.info("Saving tokenizer to: %s", output_dir)
            self.tokenizer.save_pretrained(output_dir)
