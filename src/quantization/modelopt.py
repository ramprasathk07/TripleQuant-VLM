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
from torch.utils.data import DataLoader

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
        """Run NVIDIA ModelOpt quantization with the configured scheme.

        Orchestrates: calibration dataloader preparation, quant_cfg building
        (scheme -> modelopt default + group_size/ignore patching), quantization
        with calibration forward loop, and model export.
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
        via apply_chat_template, then wraps in a torch DataLoader (batch_size=1).

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
            # Build messages from the dataset (assumes 'messages' field).
            # VLM templates require list-of-parts content; LLM templates expect plain string.
            messages = []
            for msg in example["messages"]:
                content = (
                    [{"type": "text", "text": msg["content"]}]
                    if is_vlm
                    else msg["content"]
                )
                messages.append({"role": msg["role"], "content": content})

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
        ds.set_format(type="torch")
        self._calib_dataloader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

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

    # Scheme -> modelopt default cfg (non-AWQ path).
    _SCHEME_CFG_MAP = {
        "W4A16": "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
        "W8A16": "INT8_WEIGHT_ONLY_CFG",
        "W8A8": "INT8_DEFAULT_CFG",
        "W8A8_SMOOTH": "INT8_SMOOTHQUANT_CFG",
        "FP8": "FP8_DEFAULT_CFG",
        "FP8_DYNAMIC": "FP8_PER_CHANNEL_PER_TOKEN_CFG",
        "FP8_KV": "FP8_KV_CFG",
        "NVFP4": "NVFP4_DEFAULT_CFG",
        "MXFP4": "MXFP4_DEFAULT_CFG",
        "MXFP6": "MXFP6_DEFAULT_CFG",
        "MXFP8": "MXFP8_DEFAULT_CFG",
        "MXINT8": "MXINT8_DEFAULT_CFG",
    }

    # AWQ variants of the above.
    _SCHEME_AWQ_CFG_MAP = {
        "W4A16": "INT4_AWQ_CFG",
        "NVFP4": "NVFP4_AWQ_LITE_CFG",
    }

    def _build_modelopt_config(self) -> Dict[str, Any]:
        """Build ModelOpt 0.44 list-based quant_cfg from SchemeConfig.

        ModelOpt 0.44+ uses ``quant_cfg`` as an ordered list of
        ``{"quantizer_name": "<glob>", ...}`` entries. We start from the
        backend-supplied default for the requested scheme, patch the weight
        quantizer's ``block_sizes`` to honor ``group_size``, and append
        ``enable=False`` entries for the merged ignore list.

        Returns:
            Dict with keys ``quant_cfg`` (list) and ``algorithm`` (str or dict).

        Raises:
            ValueError: If scheme is not supported by ModelOpt backend.
        """
        scheme = self.config.scheme.scheme
        method = self.config.method
        group_size = self.config.scheme.group_size

        base = self._select_default_cfg(scheme, method)
        # Deep copy: list of dicts, each dict may contain a nested "cfg".
        quant_list = []
        for entry in base["quant_cfg"]:
            new_entry = dict(entry)
            if "cfg" in new_entry:
                new_entry["cfg"] = dict(new_entry["cfg"])
                if "block_sizes" in new_entry["cfg"]:
                    new_entry["cfg"]["block_sizes"] = dict(new_entry["cfg"]["block_sizes"])
            quant_list.append(new_entry)

        # Patch block_sizes (group_size equivalent) on weight quantizer.
        if group_size and group_size > 0:
            for entry in quant_list:
                if entry.get("quantizer_name") == "*weight_quantizer" and "cfg" in entry:
                    bs = entry["cfg"].get("block_sizes")
                    if isinstance(bs, dict) and "-1" in bs:
                        bs["-1"] = group_size

        # Append ignore patterns as enable=false entries.
        for pat in self._merged_ignore():
            glob = self._regex_to_glob(pat)
            quant_list.append({"quantizer_name": glob, "enable": False})

        quant_cfg = {
            "quant_cfg": quant_list,
            "algorithm": base.get("algorithm"),
        }

        logger.info("ModelOpt scheme=%s method=%s algorithm=%s ignore=%d",
                    scheme, method, quant_cfg["algorithm"], len(self._merged_ignore()))
        return quant_cfg

    def _select_default_cfg(self, scheme: str, method: str) -> Dict[str, Any]:
        """Resolve scheme + method to a modelopt default config object.

        Prefers an AWQ variant when ``method == 'awq'`` and one exists for the scheme.

        Raises:
            ValueError: If scheme has no default mapping in modelopt.
        """
        if method == "awq" and scheme in self._SCHEME_AWQ_CFG_MAP:
            cfg_name = self._SCHEME_AWQ_CFG_MAP[scheme]
        elif scheme in self._SCHEME_CFG_MAP:
            cfg_name = self._SCHEME_CFG_MAP[scheme]
        else:
            raise ValueError(
                f"Unsupported scheme '{scheme}' (method='{method}') for backend='modelopt'. "
                f"Supported schemes: {sorted(self._SCHEME_CFG_MAP)}"
            )
        if not hasattr(mtq, cfg_name):
            raise ValueError(
                f"modelopt build missing '{cfg_name}' — installed modelopt may be too old."
            )
        return getattr(mtq, cfg_name)

    @staticmethod
    def _regex_to_glob(pattern: str) -> str:
        """Convert our internal ignore pattern to modelopt's glob form.

        Strips leading ``re:`` and ``.*`` markers, wraps with ``*`` on both sides.
        Examples:
            ``lm_head``           -> ``*lm_head*``
            ``re:.*visual.*``     -> ``*visual*``
        """
        clean = pattern.replace("re:", "").strip()
        while clean.startswith(".*"):
            clean = clean[2:]
        while clean.endswith(".*"):
            clean = clean[:-2]
        clean = clean.strip(".")
        return f"*{clean}*" if clean else "*"

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
        """Export a deployable checkpoint via modelopt's HF export, not save_pretrained.

        Plain ``save_pretrained`` on an mtq-quantized model stores simulated-quant
        (bf16) weights and drops the quantizer state entirely — the result loads as
        ordinary bf16 in transformers (bit-identical PPL to baseline) and vLLM's
        ``--quantization modelopt`` loader refuses it ("Cannot find the config file
        for modelopt"). ``export_hf_checkpoint`` writes the real artifact: compressed
        weights + scale tensors + ``hf_quant_config.json`` that serving engines read.

        Falls back to the base (fake-quant) save with a loud warning when export
        doesn't support the scheme (e.g. simulated MX formats on this hardware),
        so quantize runs still produce *something* inspectable.
        """
        from pathlib import Path

        model_name = self.model_id.split('/')[-1]
        subfolder = f"{model_name}-{self.config.backend}-{self.config.method}-{self.config.scheme.scheme}"
        final_dir = Path(output_dir) / subfolder

        if weights:
            try:
                from modelopt.torch.export import export_hf_checkpoint
                # Export reads quantizer state off the live module; run it on GPU
                # (scale/amax buffers live where calibration ran).
                if torch.cuda.is_available() and next(self.model.parameters()).device.type != "cuda":
                    self.model.to("cuda")
                logger.info("Exporting deployable ModelOpt checkpoint -> %s", final_dir)
                export_hf_checkpoint(self.model, export_dir=str(final_dir))
            except Exception:
                logger.warning(
                    "export_hf_checkpoint failed for scheme %s — falling back to "
                    "fake-quant save_pretrained (NOT loadable by serving engines).",
                    self.config.scheme.scheme, exc_info=True,
                )
                super().save(output_dir, weights=True)
                return

        if self.processor is not None:
            self.processor.save_pretrained(str(final_dir))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(final_dir))
        logger.info("ModelOpt checkpoint exported -> %s", final_dir)