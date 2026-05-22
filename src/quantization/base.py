"""
Abstract base class for all quantizers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.schemas import QuantizeConfig

logger = logging.getLogger(__name__)


class BaseQuantizer(ABC):
    def __init__(self, config: QuantizeConfig) -> None:
        self.config = config
        self.model_id: str = config.model.model_id
        self.torch_dtype = getattr(torch, config.model.torch_dtype)
        self.device_map: str = config.model.device_map
        self.trust_remote_code: bool = config.model.trust_remote_code

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load model and processor/tokenizer from HuggingFace hub or local path."""
        logger.info("Loading model: %s (dtype=%s, device_map=%s)",
                    self.model_id, self.config.model.torch_dtype, self.device_map)

        # Use config.model.model_type to decide (not self.model which is None)
        if self.config.model.model_type == "vlm":
            self._load_vlm()
        else:
            self._load_llm()

    def _load_vlm(self) -> None:
        # Lazy import to avoid torchvision issues when not used
        from transformers import AutoProcessor, AutoModelForImageTextToText

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
    
    @abstractmethod
    def quantize(self) -> None:
        """Run quantization on *dataset* and save weights to *output_dir*."""

    # def save(self, output_dir: str, weights: bool = True) -> None:
    #     """Save model weights and/or processor/tokenizer to *output_dir*."""
    #     if weights:
    #         logger.info("Moving model to CPU before saving …")
    #         self.model.to("cpu")

    #         if hasattr(self.model, "hf_device_map"):
    #             del self.model.hf_device_map
            
    #         logger.info("Saving quantized model weights to: %s", output_dir)
    #         self.model.save_pretrained(
    #             output_dir,
    #             save_compressed=self.config.output.save_compressed,
    #         )

    #     if self.processor is not None:
    #         logger.info("Saving processor to: %s", output_dir)
    #         self.processor.save_pretrained(output_dir)

    #     if self.tokenizer is not None:
    #         logger.info("Saving tokenizer to: %s", output_dir)
    #         self.tokenizer.save_pretrained(output_dir)


    def save(self, output_dir: str, weights: bool = True) -> None:
        """Save model with a descriptive name: {model_name}-{backend}-{method}-{scheme}."""
        # Build a descriptive subfolder name
        model_name = self.model_id.split('/')[-1]  # e.g., TinyLlama-1.1B-Chat-v1.0
        backend = self.config.backend
        method = self.config.method
        scheme = self.config.scheme.scheme
        subfolder = f"{model_name}-{backend}-{method}-{scheme}"
        
        from pathlib import Path
        final_dir = Path(output_dir) / subfolder
        final_dir_str = str(final_dir)
        logger.info(f"Saving quantized model to: {final_dir_str}")

        if weights:
            logger.info("Moving model to CPU before saving …")
            self.model.to("cpu")
            if hasattr(self.model, "hf_device_map"):
                del self.model.hf_device_map
            self.model.save_pretrained(
                final_dir_str,
                save_compressed=self.config.output.save_compressed,
            )

        if self.processor is not None:
            logger.info("Saving processor to: %s", final_dir_str)
            self.processor.save_pretrained(final_dir_str)

        if self.tokenizer is not None:
            logger.info("Saving tokenizer to: %s", final_dir_str)
            self.tokenizer.save_pretrained(final_dir_str)