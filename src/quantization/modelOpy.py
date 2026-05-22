# quantization/modelopt_quantizer.py
from __future__ import annotations
import logging
import torch
from typing import Dict, Any, List, Optional, Callable
from datasets import load_dataset, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

from src.config.schemas import QuantizeConfig
from .base import BaseQuantizer

logger = logging.getLogger(__name__)


class ModelOptQuantizer(BaseQuantizer):
    """NVIDIA ModelOpt quantizer supporting NVFP4, MXFP4, FP8, INT4, INT8, with advanced tuning."""

    def __init__(self, config: QuantizeConfig) -> None:
        super().__init__(config)
        self._calib_dataloader = None

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------
    def quantize(self) -> None:
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

    # -------------------------------------------------------------------------
    # Calibration data loading (multimodal)
    # -------------------------------------------------------------------------
    def _prepare_calibration_dataloader(self) -> None:
        """Load and preprocess calibration dataset, supporting both text and images."""
        cal = self.config.calibration
        tokenizer = self.processor if self.processor else self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer/processor available for calibration.")

        # Load raw dataset (assumes 'LLM' configuration or use default)
        ds = load_dataset(cal.dataset_name, name="LLM", split=f"train[:{cal.num_samples}]")
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
        """Collate a batch of samples (single sample only for VLM stability)."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        collated = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "pixel_values" in batch[0]:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            collated["pixel_values"] = pixel_values
        return collated

    # -------------------------------------------------------------------------
    # Vision tower detection and ignore pattern merging
    # -------------------------------------------------------------------------
    def _is_vlm(self) -> bool:
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

    def _merged_ignore_patterns(self) -> List[str]:
        """Return list of module name patterns to exclude from quantization."""
        user_ignore = self.config.scheme.ignore
        vision_ignore = self._get_vision_ignore_patterns()
        merged = list(user_ignore)
        for pat in vision_ignore:
            if pat not in merged:
                merged.append(pat)
        if vision_ignore:
            logger.info("Auto‑ignoring vision modules: %s", vision_ignore)
        return merged

    # -------------------------------------------------------------------------
    # Build ModelOpt quantization config with advanced options
    # -------------------------------------------------------------------------
    def _build_modelopt_config(self) -> Dict[str, Any]:
        """Convert our SchemeConfig into a ModelOpt quant_cfg dict, respecting observer, per_channel, etc."""
        scheme = self.config.scheme.scheme
        group_size = self.config.scheme.group_size
        symmetric = self.config.scheme.symmetric
        observer = getattr(self.config.scheme, "observer", "mse")
        per_channel = getattr(self.config.scheme, "per_channel", False)
        dynamic_activations = getattr(self.config.scheme, "dynamic_activations", False)

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
        """Set quantize=False for all patterns in the merged ignore list."""
        ignore_patterns = self._merged_ignore_patterns()
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
        """Custom config for NVFP4 (4‑bit float, 2 exponent bits)."""
        return {
            "algorithm": observer,
            "quant_cfg": {
                ".*weight": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": symmetric,
                    "group_size": group_size,
                    "per_channel": per_channel,
                    "exponent_bits": 2,
                    "mantissa_bits": 1,
                }
            },
            "quantize_input": False,
            "quantize_output": False,
        }

    def _get_mxfp4_cfg(self, group_size: int, symmetric: bool, observer: str, per_channel: bool) -> Dict[str, Any]:
        """Custom config for MXFP4 (Microscaling FP4 with block scaling)."""
        return {
            "algorithm": observer,
            "quant_cfg": {
                ".*weight": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": symmetric,
                    "group_size": group_size,
                    "per_channel": per_channel,
                    "block_scaling": True,
                    "exponent_bits": 2,
                    "mantissa_bits": 1,
                }
            },
            "quantize_input": False,
            "quantize_output": False,
        }

    # -------------------------------------------------------------------------
    # Apply quantization using ModelOpt
    # -------------------------------------------------------------------------
    def _apply_quantization(self, quant_cfg: Dict[str, Any]) -> None:
        """Run mtq.quantize with forward loop for calibration."""
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

    # -------------------------------------------------------------------------
    # Native save support (ModelOpt + compressed tensors)
    # -------------------------------------------------------------------------
    def save(self, output_dir: str, weights: bool = True) -> None:
        """Save quantized model natively. Optionally exports to TensorRT Engine."""
        if weights:
            self.model.to("cpu")
            if hasattr(self.model, "hf_device_map"):
                del self.model.hf_device_map

            # Attempt to use ModelOpt's export if available (for INT8/FP8)
            try:
                from modelopt.torch.export import export_tensorrt_engine
                # This exports to a TensorRT engine; but we also want the HuggingFace format.
                # For simplicity, we first save with save_pretrained.
                self.model.save_pretrained(output_dir, save_compressed=self.config.output.save_compressed)
                logger.info("Saved HuggingFace model to %s", output_dir)
                # Optionally export to TensorRT for deployment
                # export_tensorrt_engine(self.model, output_dir + "/engine")
            except ImportError:
                # Fallback: standard save_pretrained (works for INT4/INT8)
                self.model.save_pretrained(output_dir, save_compressed=self.config.output.save_compressed)
                logger.info("Saved quantized model (no TensorRT export) to %s", output_dir)
        # Save processor/tokenizer
        if self.processor and self.config.output.save_processor:
            self.processor.save_pretrained(output_dir)
        if self.tokenizer and self.config.output.save_processor:
            self.tokenizer.save_pretrained(output_dir)