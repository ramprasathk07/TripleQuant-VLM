"""Abstract base class for all quantizers.

Defines the interface and shared utilities for quantization backends (llm_compressor,
modelopt, future: turboquant). Handles model and processor/tokenizer loading for both
LLMs and vision-language models (VLMs), with automatic VLM detection and vision-module
exclusion from quantization.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.schemas import QuantizeConfig

logger = logging.getLogger(__name__)


class BaseQuantizer(ABC):
    """Abstract base class for quantization backends.

    Subclasses implement the quantize() method for backend-specific algorithms
    (AWQ, GPTQ, NVIDIA ModelOpt, etc.). Provides shared infrastructure for model
    loading, dataset preprocessing, and result export.

    Attributes:
        config: QuantizeConfig instance controlling all aspects of quantization.
        model_id: Hugging Face model identifier.
        torch_dtype: PyTorch dtype (e.g., torch.bfloat16).
        device_map: Device placement strategy for loading (e.g., "auto").
        model: Loaded transformer model (set by load_model()).
        processor: VLM image/text processor (if applicable).
        tokenizer: LLM tokenizer (if applicable).
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

    def load_model(self) -> None:
        """Load model and associated tokenizer/processor from Hugging Face.

        Routes to _load_vlm() or _load_llm() based on config.model.model_type.
        """
        logger.info("Loading model: %s (dtype=%s, device_map=%s)",
                    self.model_id, self.config.model.torch_dtype, self.device_map)

        # Use config.model.model_type to decide (not self.model which is None)
        if self.config.model.model_type == "vlm":
            self._load_vlm()
        else:
            self._load_llm()

    def _load_vlm(self) -> None:
        """Load a vision-language model + its AutoProcessor.

        Tries auto-classes in order, then the config's explicit architecture class.
        Needed because some VLMs (e.g. HunyuanOCR / HunYuanVLConfig) are not in the
        AutoModelForImageTextToText mapping even though they ship a
        ``*ForConditionalGeneration`` class.
        """
        from transformers import AutoProcessor

        processor_kwargs: dict = {}
        if self.config.model.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.model.min_pixels
        if self.config.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.model.max_pixels

        logger.info("Detected VLM — loading AutoProcessor + VLM model")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            **processor_kwargs,
        )
        self.model = self._from_pretrained_vlm()

    def _from_pretrained_vlm(self):
        """Load the VLM weights via a fallback chain of model classes."""
        import transformers
        from transformers import (
            AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM,
        )

        kw = dict(
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )

        last_err: Exception | None = None
        for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
            try:
                model = loader.from_pretrained(self.model_id, **kw)
                logger.info("Loaded VLM via %s", loader.__name__)
                return model
            except (ValueError, KeyError) as e:
                last_err = e
                logger.warning("%s could not load this model — trying next loader.",
                               loader.__name__)

        # Last resort: the explicit architecture class named in the config
        # (e.g. HunYuanVLForConditionalGeneration).
        try:
            cfg = AutoConfig.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code,
            )
            arch = (getattr(cfg, "architectures", None) or [None])[0]
            cls = getattr(transformers, arch, None) if arch else None
            if cls is not None:
                model = cls.from_pretrained(self.model_id, **kw)
                logger.info("Loaded VLM via explicit class %s", arch)
                return model
            logger.error("Architecture '%s' not importable from transformers.", arch)
        except Exception as e:  # noqa: BLE001
            last_err = e

        raise RuntimeError(
            f"Could not load VLM '{self.model_id}' via any loader."
        ) from last_err

    def _load_llm(self) -> None:
        """Load language model with AutoTokenizer and AutoModelForCausalLM."""
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
        """Run quantization on calibration dataset and save to output directory.

        Subclass implementations load calibration data, apply backend-specific
        quantization (AWQ, GPTQ, ModelOpt, etc.), and invoke save() for export.
        """

    def _is_vlm(self) -> bool:
        """Detect if the loaded model is a vision-language model using heuristics.

        Checks for vision_config/vision_tower attributes and architecture names
        containing 'vl', 'vision', or 'multi_modal'.

        Returns:
            True if model appears to be a VLM, False otherwise.
        """
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return False
        if hasattr(cfg, "vision_config") or hasattr(cfg, "vision_tower"):
            return True
        arch = getattr(cfg, "architectures", [None])[0]
        if arch and any(v in arch.lower() for v in ("vl", "vision", "multi_modal")):
            return True
        return False

    def _get_vision_ignore_patterns(self) -> List[str]:
        """Regex patterns for vision tower modules to exclude from quantization.

        Returns empty list for non-VLM models; returns patterns for common vision
        modules (visual tower, vision model, mm_projector, etc.) for VLMs.

        Returns:
            List of regex patterns (re:* format) for vision modules.
        """
        if not self._is_vlm():
            return []
        return [
            "re:.*visual.*",
            "re:.*vision_tower.*",
            "re:.*vision_model.*",
            "re:.*mm_projector.*",
            "re:.*multi_modal_projector.*",
            "re:.*merger.*",
            "re:.*patch_embed.*",
        ]

    def _merged_ignore(self) -> List[str]:
        """Merge user-provided ignore patterns with auto-detected vision module patterns.

        Returns:
            Combined list of all ignore patterns to exclude from quantization.
        """
        user_ignore = self.config.scheme.ignore
        vision_ignore = self._get_vision_ignore_patterns()
        merged = list(user_ignore)
        for pat in vision_ignore:
            if pat not in merged:
                merged.append(pat)
        if vision_ignore:
            logger.info("Auto-ignoring vision modules: %s", vision_ignore)
        return merged

    def save(self, output_dir: str, weights: bool = True) -> None:
        """Save quantized model, processor, and tokenizer to a descriptive subfolder.

        Creates subfolder: {model_name}-{backend}-{method}-{scheme}. Optionally
        saves model weights in compressed format; always saves processor/tokenizer if available.

        Args:
            output_dir: Root output directory path.
            weights: If True, save model weights; if False, skip model saving.
        """
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