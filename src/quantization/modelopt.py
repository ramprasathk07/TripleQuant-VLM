"""NVIDIA ModelOpt quantization backend supporting FP8, NVFP4, MXFP4, INT variants.

Integrates nvidia-modelopt library for low-precision quantization with advanced
observer strategies (MSE, minmax, percentile), per-channel granularity, and
dynamic activation support. Automatically excludes vision modules for VLMs.
"""
from __future__ import annotations
import logging
import torch
from typing import Dict, Any, List
from datasets import load_dataset

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

from src.config.schemas import QuantizeConfig
from .base import BaseQuantizer

logger = logging.getLogger(__name__)


class ModelOptQuantizer(BaseQuantizer):
    """NVIDIA ModelOpt quantizer.

    Supports FP8, NVFP4, MXFP4, INT4, INT8 quantization schemes with configurable
    observer strategies, per-channel quantization, dynamic activations, and AWQ
    integration. Multimodal-ready with automatic vision module exclusion.

    Attributes:
        _calib_dataloader: Prepared dataloader for calibration forward passes.
    """

    def __init__(self, config: QuantizeConfig) -> None:
        super().__init__(config)
        self._calib_dataloader = None

    def quantize(self) -> None:
        """Run NVIDIA ModelOpt quantization with configured scheme and observer.

        Orchestrates: calibration dataloader preparation, quantization config
        building (scheme mapping + observer selection), apply_quantization with
        forward loop, and model export.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Starting ModelOpt quantization with backend=modelopt")
        # 1. Prepare calibration dataloader (handles text+images)
        self._prepare_calibration_dataloader()
        # 2. Build ModelOpt quant config from our SchemeConfig
        quant_cfg = self._build_modelopt_config()
        # 3. Apply quantization
        self._apply_quantization(quant_cfg)
        # 4. Save the quantized model (native ModelOpt export if possible)
        self.save(str(self.config.output.output_dir))
        logger.info("Quantization finished and model saved to %s", self.config.output.output_dir)

    def _prepare_calibration_dataloader(self) -> None:
        """Load and preprocess calibration dataset for ModelOpt forward loop.

        Assumes chat-format datasets with optional image fields for VLMs.
        Preprocesses to input_ids, attention_mask, and optionally pixel_values
        via apply_chat_template. Prepares dataloader via get_dataset_dataloader.

        Raises:
            ValueError: If no tokenizer/processor is available.
        """
        cal = self.config.calibration
        tokenizer = self.processor if self.processor else self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer/processor available for calibration.")

        split = cal.split if "[:" in cal.split else f"{cal.split}[:{cal.num_samples}]"
        ds = load_dataset(cal.dataset_name, split=split)
        ds = ds.shuffle(seed=cal.seed)

        is_vlm = self._is_vlm()

        def preprocess_fn(example):
            # Build messages from the dataset (assumes 'messages' field)
            messages = []
            for msg in example["messages"]:
                messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })

            # For VLM, handle images if present
            if is_vlm and cal.image_field and cal.image_field in example:
                images = example[cal.image_field]
                if not isinstance(images, list):
                    images = [images]
                processed = tokenizer.apply_chat_template(
                    messages,
                    images=images,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=cal.max_seq_len,
                    tokenize=True,
                    add_special_tokens=False,
                    return_dict=True,
                    add_generation_prompt=False,
                )
            else:
                processed = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=cal.max_seq_len,
                    tokenize=True,
                    add_special_tokens=False,
                    return_dict=True,
                    add_generation_prompt=False,
                )
            # ModelOpt expects input_ids, attention_mask, and optionally pixel_values
            result = {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed.get("attention_mask", torch.ones_like(processed["input_ids"])).squeeze(0)
            }
            if "pixel_values" in processed:
                result["pixel_values"] = processed["pixel_values"]
            return result

        ds = ds.map(preprocess_fn, batched=False, remove_columns=ds.column_names)
        self._calib_dataloader = get_dataset_dataloader(ds, batch_size=1, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of preprocessed samples into tensor dictionary.

        Stacks input_ids, attention_mask, and pixel_values (if present).

        Args:
            batch: List of preprocessed examples with input_ids, attention_mask, optional pixel_values.

        Returns:
            Dict with stacked tensors: input_ids, attention_mask, and optionally pixel_values.
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        collated = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "pixel_values" in batch[0]:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            collated["pixel_values"] = pixel_values
        return collated

    def _build_modelopt_config(self) -> Dict[str, Any]:
        """Build ModelOpt quantization config from SchemeConfig and advanced options.

        Maps quantization scheme (W4A16, W8A8, FP8, NVFP4, etc.) to ModelOpt
        defaults, applies observer strategy, per_channel/dynamic_activation tuning,
        excludes vision modules via merged ignore patterns, and handles AWQ if requested.

        Returns:
            Dict with ModelOpt quant_cfg structure (algorithm, quant_cfg sub-dict, etc.).

        Raises:
            ValueError: If scheme is not supported by ModelOpt backend.
        """
        scheme = self.config.scheme.scheme
        group_size = self.config.scheme.group_size
        symmetric = self.config.scheme.symmetric
        observer = self.config.scheme.observer
        per_channel = self.config.scheme.per_channel
        dynamic_activations = self.config.scheme.dynamic_activations

        # Map our observer strings to ModelOpt algorithm names
        observer_map = {
            "mse": "mse",
            "minmax": "minmax",
            "maxabs": "max_abs",
            "percentile": "percentile",
        }
        algorithm = observer_map.get(observer, "max_abs")

        # Base config structure that ModelOpt expects
        quant_cfg = {
            "algorithm": algorithm,
            "quantize_input": "A8" in scheme or "FP8" in scheme,
            "quantize_output": False,
        }

        # ---- Precision mapping ----
        if scheme in ("W4A16", "W4A16_ASYM"):
            quant_cfg = mtq.INT4_DEFAULT_CFG.copy()
            quant_cfg["quant_cfg"] = {
                ".*weight": {
                    "num_bits": 4,
                    "group_size": group_size,
                    "symmetric": symmetric and "ASYM" not in scheme,
                    "observer": algorithm,
                    "per_channel": per_channel,
                }
            }
        elif scheme in ("W8A8", "W8A8_ASYM"):
            quant_cfg = mtq.INT8_DEFAULT_CFG.copy()
            quant_cfg["quant_cfg"][".*weight"]["group_size"] = group_size
            quant_cfg["quant_cfg"][".*weight"]["symmetric"] = symmetric and "ASYM" not in scheme
            quant_cfg["quant_cfg"][".*weight"]["observer"] = algorithm
            quant_cfg["quant_cfg"][".*weight"]["per_channel"] = per_channel
            if dynamic_activations:
                quant_cfg["quant_cfg"][".*input"]["dynamic"] = True
            if "ASYM" in scheme:
                quant_cfg["quant_cfg"][".*input"]["symmetric"] = False
        elif scheme == "W8A16":
            quant_cfg = mtq.INT8_SMOOTHQUANT_CFG.copy() if hasattr(mtq, "INT8_SMOOTHQUANT_CFG") else mtq.INT8_DEFAULT_CFG.copy()
            quant_cfg["quant_cfg"] = {
                ".*weight": {
                    "num_bits": 8,
                    "symmetric": symmetric,
                    "observer": algorithm,
                    "per_channel": per_channel,
                }
            }
        elif scheme == "FP8":
            quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
            # FP8 doesn't use group_size
        elif scheme == "FP8_DYNAMIC":
            quant_cfg = mtq.FP8_DYNAMIC_CFG.copy()
        elif scheme == "FP8_BLOCK":
            if hasattr(mtq, "FP8_BLOCK_CFG"):
                quant_cfg = mtq.FP8_BLOCK_CFG.copy()
            else:
                quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
            if self.config.scheme.block_size:
                quant_cfg["quant_cfg"][".*weight"]["block_size"] = self.config.scheme.block_size
        elif scheme == "NVFP4":
            quant_cfg = self._get_nvfp4_cfg(group_size, symmetric, algorithm, per_channel)
        elif scheme == "MXFP4":
            quant_cfg = self._get_mxfp4_cfg(group_size, symmetric, algorithm, per_channel)
        else:
            raise ValueError(f"Unsupported scheme for ModelOpt: {scheme}")

        # Apply AWQ if requested
        if self.config.method == "awq":
            quant_cfg["algorithm"] = "awq"
            if self.config.awq and self.config.awq.duo_scaling:
                quant_cfg["awq_params"] = {"enable_duo_scaling": True}

        # Exclude vision modules from quantization
        quant_cfg = self._exclude_vision_modules(quant_cfg)

        logger.info("ModelOpt quant config: algorithm=%s, per_channel=%s, dynamic_activations=%s",
                    algorithm, per_channel, dynamic_activations)
        return quant_cfg

    def _exclude_vision_modules(self, quant_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Set quantize=False in quant_cfg for all merged ignore patterns.

        Args:
            quant_cfg: ModelOpt quantization config dict to modify in-place.

        Returns:
            Modified quant_cfg with vision modules marked quantize=False.
        """
        ignore_patterns = self._merged_ignore()
        if not ignore_patterns:
            return quant_cfg
        # Ensure quant_cfg has a "quant_cfg" subsection
        if "quant_cfg" not in quant_cfg:
            quant_cfg["quant_cfg"] = {}
        for pattern in ignore_patterns:
            # Convert re:.*something to pattern without re: prefix if needed
            clean_pattern = pattern.replace("re:", "").replace(".*", "")
            quant_cfg["quant_cfg"][clean_pattern] = {"quantize": False}
        return quant_cfg

    def _get_nvfp4_cfg(self, group_size: int, symmetric: bool, observer: str, per_channel: bool) -> Dict[str, Any]:
        """Build NVFP4 quantization config from ModelOpt defaults with custom overrides.

        Args:
            group_size: Grouped quantization granularity.
            symmetric: Symmetric or asymmetric quantization.
            observer: Calibration observer strategy (not used for NVFP4 but kept for API consistency).
            per_channel: Per-channel vs. per-group quantization.

        Returns:
            NVFP4 ModelOpt config dict.
        """
        cfg = mtq.NVFP4_DEFAULT_CFG.copy()
        weight_cfg = cfg["quant_cfg"][".*weight"]
        weight_cfg["group_size"] = group_size
        weight_cfg["symmetric"] = symmetric
        weight_cfg["per_channel"] = per_channel
        return cfg

    def _get_mxfp4_cfg(self, group_size: int, symmetric: bool, observer: str, per_channel: bool) -> Dict[str, Any]:
        """Build MXFP4 quantization config from ModelOpt defaults with custom overrides.

        Args:
            group_size: Grouped quantization granularity.
            symmetric: Symmetric or asymmetric quantization.
            observer: Calibration observer strategy (not used for MXFP4 but kept for API consistency).
            per_channel: Per-channel vs. per-group quantization.

        Returns:
            MXFP4 ModelOpt config dict.
        """
        cfg = mtq.MXFP4_DEFAULT_CFG.copy()
        weight_cfg = cfg["quant_cfg"][".*weight"]
        weight_cfg["group_size"] = group_size
        weight_cfg["symmetric"] = symmetric
        weight_cfg["per_channel"] = per_channel
        return cfg

    def _apply_quantization(self, quant_cfg: Dict[str, Any]) -> None:
        """Apply ModelOpt quantization with calibration forward loop.

        Ensures model is on GPU, defines forward_loop for calibration using
        prepared dataloader, calls mtq.quantize(), and moves model back to CPU.

        Args:
            quant_cfg: ModelOpt quantization config dict.

        Raises:
            Exception: If mtq.quantize fails (wrapped with logging).
        """
        # Move model to GPU
        if next(self.model.parameters()).device.type != "cuda":
            self.model.to("cuda")

        def forward_loop(model: torch.nn.Module):
            with torch.no_grad():
                for batch in self._calib_dataloader:
                    input_ids = batch["input_ids"].to("cuda")
                    attention_mask = batch["attention_mask"].to("cuda")
                    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                    if "pixel_values" in batch:
                        kwargs["pixel_values"] = batch["pixel_values"].to("cuda")
                    model(**kwargs)

        logger.info("Applying ModelOpt quantization with config:\n%s", quant_cfg)
        try:
            mtq.quantize(self.model, quant_cfg, forward_loop=forward_loop)
        except Exception as e:
            logger.error("ModelOpt quantization failed: %s", e, exc_info=True)
            raise

        # After quantization, move back to CPU for saving
        self.model.to("cpu")

    def save(self, output_dir: str, weights: bool = True) -> None:
        """Save quantized model via BaseQuantizer with descriptive subfolder naming.

        Args:
            output_dir: Root output directory path.
            weights: If True, save model weights; if False, skip model saving.
        """
        super().save(output_dir, weights=weights)