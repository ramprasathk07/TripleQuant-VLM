"""LLMCompressor quantization backend supporting AWQ, GPTQ, PTQ, and SmoothQuant.

Integrates llm-compressor library for multimodal quantization with support for
both text-only LLMs and vision-language models (VLMs). Handles automatic dataset
format detection (chat vs. image-text) and per-batch calibration.
"""
from __future__ import annotations
import logging
import torch
from typing import Dict, Any, List
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from compressed_tensors.quantization import preset_name_to_scheme
from src.config.schemas import QuantizeConfig, _MODELOPT_ONLY_SCHEMES
from .base import BaseQuantizer

logger = logging.getLogger(__name__)


class LLMCompressorQuantizer(BaseQuantizer):
    """LLMCompressor quantization backend.

    Supports AWQ (activation-aware), GPTQ (gradient-based), and PTQ (post-training)
    quantization with optional SmoothQuant activation quantization. Handles both
    text and multimodal calibration datasets with automatic format detection and
    preprocessing for chat vs. image-text formats.
    """

    def __init__(self, config: QuantizeConfig):
        super().__init__(config)
        self._profile = None

    def quantize(self) -> None:
        """Run llmcompressor oneshot quantization with configured scheme and modifiers.

        Orchestrates: dataset loading (with format auto-detection), recipe building
        (modifiers for AWQ/GPTQ/PTQ + SmoothQuant), oneshot calibration, and model export.
        """
        logger.info("Starting LLMCompressor quantization with backend=llm_compressor")
        # 1. Load calibration dataset (handles text and images)
        dataset = self._load_calibration_dataset()
        # 2. Build recipe from config
        recipe = self._build_recipe()
        # Log the recipe for debugging
        logger.debug(f"Built recipe with {len(recipe)} modifiers")
        # 3. Run oneshot
        oneshot(
            model=self.model,
            processor=self.processor or self.tokenizer,
            recipe=recipe,
            dataset=dataset,
            max_seq_length=self.config.calibration.max_seq_len,
            num_calibration_samples=self.config.calibration.num_samples,
            data_collator=self._data_collator,
        )
        # 4. Save the quantized model
        self.save(str(self.config.output.output_dir))

        # 5. Export to onnx
        
        logger.info("Quantization finished and model saved to %s", self.config.output.output_dir)

    def _load_calibration_dataset(self) -> Dataset:
        """Load and preprocess calibration dataset with format auto-detection.

        Supports chat-format datasets (ultrachat_200k style) and image-text pairs
        (LaTeX_OCR style). Auto-detects format and column names; handles VLM-specific
        image preprocessing (RGB conversion, minimum size enforcement).

        Returns:
            Preprocessed Dataset with tokenized inputs and pixel_values (if VLM).

        Raises:
            ValueError: If dataset split, image column, or text column cannot be resolved.
        """
        cal = self.config.calibration
        split = getattr(cal, 'split', 'test')
        if "[:" not in split:
            split = f"{split}[:{cal.num_samples}]"

        # Multi-config hub datasets (e.g. linxy/LaTeX_OCR) require a config name.
        subset = getattr(cal, 'subset', None)
        try:
            ds = load_dataset(cal.dataset_name, subset, split=split)
        except ValueError as e:
            from datasets import get_dataset_config_names, get_dataset_split_names
            try:
                configs = get_dataset_config_names(cal.dataset_name)
            except Exception:
                configs = []
            try:
                splits = get_dataset_split_names(cal.dataset_name, subset)
            except Exception:
                splits = []
            raise ValueError(
                f"Could not load '{cal.dataset_name}' (config={subset!r}, split='{split}'). "
                f"Available configs: {configs}. Available splits: {splits}. "
                f"Set 'calibration.subset' and/or 'calibration.split' in your config."
            ) from e

        ds = ds.shuffle(seed=cal.seed)
        tokenizer = self.processor if self.processor else self.tokenizer
        is_vlm = self._is_vlm()

        # ── Detect dataset format ─────────────────────────────────────────────
        dataset_format = getattr(cal, 'dataset_format', 'chat')
        # 'chat'          → expects a 'messages' column (ultrachat_200k style)
        # 'image_text'    → expects separate image + text columns (LaTeX_OCR style)

        logger.info(
            "Calibration dataset: %s | format: %s | columns: %s",
            cal.dataset_name, dataset_format, ds.column_names
        )

        # ── Auto-detect format if not set ────────────────────────────────────
        if dataset_format == 'auto':
            if 'messages' in ds.column_names:
                dataset_format = 'chat'
                logger.info("Auto-detected dataset format: chat")
            else:
                dataset_format = 'image_text'
                logger.info("Auto-detected dataset format: image_text")

        # ── Resolve column names for image_text format ────────────────────────
        # Config can override; fall back to common names automatically.
        # If config value is set but column doesn't exist, warn and auto-detect.
        image_col   = getattr(cal, 'image_field',   None)
        text_col    = getattr(cal, 'text_field',     None)

        if image_col and image_col not in ds.column_names:
            logger.warning(
                "image_field '%s' not in dataset columns %s — switching to auto-detect",
                image_col, ds.column_names,
            )
            image_col = None

        if text_col and text_col not in ds.column_names:
            logger.warning(
                "text_field '%s' not in dataset columns %s — switching to auto-detect",
                text_col, ds.column_names,
            )
            text_col = None

        if dataset_format == 'image_text' and image_col is None:
            # auto-detect image column
            for candidate in ('image', 'img', 'pixel_values', 'image_path'):
                if candidate in ds.column_names:
                    image_col = candidate
                    break
            if image_col is None:
                raise ValueError(
                    f"Could not auto-detect image column in dataset. "
                    f"Available columns: {ds.column_names}. "
                    f"Set 'calibration.image_field' in your config."
                )
            logger.info("Auto-detected image column: '%s'", image_col)

        if dataset_format == 'image_text' and text_col is None:
            for candidate in ('text', 'formula', 'latex', 'label', 'caption', 'answer'):
                if candidate in ds.column_names:
                    text_col = candidate
                    break
            if text_col is None:
                raise ValueError(
                    f"Could not auto-detect text column in dataset. "
                    f"Available columns: {ds.column_names}. "
                    f"Set 'calibration.text_field' in your config."
                )
            logger.info("Auto-detected text column: '%s'", text_col)

        # ── Preprocessors ─────────────────────────────────────────────────────

        def preprocess_chat(example):
            """Preprocess chat-format example (messages field) for LLM or VLM."""
            messages = []
            for msg in example["messages"]:
                if is_vlm and cal.image_field and cal.image_field in example:
                    content = [{"type": "text", "text": msg["content"]}]
                else:
                    content = msg["content"]
                messages.append({"role": msg["role"], "content": content})

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
            return dict(processed)

        def preprocess_image_text(example):
            """Preprocess image-text pair (e.g., LaTeX_OCR) into chat format.

            Constructs user-assistant message pair: user turn has [image] + instruction,
            assistant turn has ground-truth text. Enforces RGB PIL format and minimum
            56x56 size to avoid Qwen2.5-VL smart_resize crashes.
            """
            from PIL import Image as PILImage

            image  = example[image_col]
            answer = str(example[text_col])

            # Ensure RGB PIL image — Qwen image processor rejects grayscale/palette/tiny.
            if not isinstance(image, PILImage.Image):
                image = PILImage.fromarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Qwen2.5-VL smart_resize factor = patch_size(14) * merge_size(2) = 28.
            # round(dim / 28) * 28 = 0 for dim < 14 → resize crash.
            # Enforce minimum 56×56 (2 patches each side).
            w, h = image.size
            if w < 56 or h < 56:
                image = image.resize((max(w, 56), max(h, 56)), PILImage.LANCZOS)

            instruction = getattr(
                cal, 'instruction_prompt',
                "Convert the image to LaTeX formula."
            )

            # Assistant content MUST be list-of-dicts, not a plain string.
            # Qwen2.5-VL processor iterates message["content"] expecting dicts with "type".
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ]

            try:
                # Do NOT pass images= kwarg: Qwen2.5-VL apply_chat_template auto-extracts
                # images from {"type":"image","image":...} in message content.
                # Passing images= explicitly causes "got multiple values" TypeError.
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
            except Exception as e:
                # Fallback: two-step — render template to text, then call processor with images.
                logger.debug("apply_chat_template failed (%s), using two-step processor() call", e)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                processed = tokenizer(
                    text=text,
                    images=[image],
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=cal.max_seq_len,
                )

            return dict(processed)

        # ── Apply the right preprocessor ─────────────────────────────────────
        if dataset_format == 'chat':
            ds = ds.map(preprocess_chat,       batched=False, remove_columns=ds.column_names)
        else:
            ds = ds.map(preprocess_image_text, batched=False, remove_columns=ds.column_names)

        return ds

    def _data_collator(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch (single sample only) into tensor dictionary.

        Converts pixel_values to bfloat16; enforces batch_size=1 for stability.

        Args:
            batch: List of preprocessed examples (length 1).

        Returns:
            Dict with input_ids, attention_mask, and optionally pixel_values as tensors.

        Raises:
            AssertionError: If batch length is not 1.
        """
        assert len(batch) == 1, "Only single‑sample batching supported for now"
        elem = batch[0]
        collated = {}
        for key, value in elem.items():
            if key == "pixel_values":
                # pixel_values may be list or tensor; convert to bfloat16 and squeeze batch dim
                collated[key] = torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
            else:
                collated[key] = torch.tensor(value)
        return collated

    def _build_quant_config(self) -> Dict[str, Any]:
        """Resolve the weight/activation quant config for the configured scheme.

        Uses compressed-tensors preset schemes as the source of truth (correct
        int/float dtype, strategy, group size and dynamic flags for every
        supported scheme: W4A16, W8A8, W4A8, FP8, FP8_DYNAMIC, NVFP4, …), then
        overlays the user-supplied knobs where they are safe for the resolved
        strategy. This is what makes every llm_compressor-supported format work
        through one code path instead of an INT-only hand-built dict.

        Returns:
            A config_groups entry: ``{"weights": {...}, "input_activations": {...}}``
            (the ``input_activations`` key is present only when the scheme quantizes
            activations).

        Raises:
            ValueError: If the scheme is modelopt-only or unknown to llm_compressor.
        """
        scheme = self.config.scheme.scheme
        targets = self.config.scheme.targets
        sc = self.config.scheme

        if scheme in _MODELOPT_ONLY_SCHEMES:
            raise ValueError(
                f"scheme='{scheme}' is modelopt-only; use backend='modelopt'."
            )
        try:
            preset = preset_name_to_scheme(scheme, targets)
        except Exception as e:
            raise ValueError(
                f"scheme='{scheme}' is not supported by llm_compressor: {e}"
            ) from e

        weights = preset.weights.model_dump(exclude_none=True)
        acts = (
            preset.input_activations.model_dump(exclude_none=True)
            if preset.input_activations is not None
            else None
        )

        # ── Overlay user knobs (only where they don't contradict the dtype/strategy) ──
        is_int = weights.get("type") == "int"

        # per_channel forces channel strategy (int weight-only schemes); drops group_size.
        if sc.per_channel and is_int:
            weights["strategy"] = "channel"
            weights.pop("group_size", None)
        # group_size override only meaningful for grouped strategies.
        elif weights.get("strategy") == "group" and sc.group_size:
            weights["group_size"] = sc.group_size

        # symmetric / observer only for INT weights — float schemes (FP8/NVFP4)
        # require symmetric=True and use their preset observer.
        if is_int:
            weights["symmetric"] = sc.symmetric
            if not weights.get("dynamic", False) and sc.observer:
                weights["observer"] = sc.observer

        # actorder is a GPTQ-only heuristic and is only valid for group strategy.
        if (
            self.config.method == "gptq"
            and is_int
            and weights.get("strategy") == "group"
        ):
            weights["actorder"] = bool(sc.actorder)

        # Each config_groups entry is a full QuantizationScheme — it carries its
        # own targets (required by the modifier's config_groups schema).
        group_config: Dict[str, Any] = {"targets": list(targets), "weights": weights}
        if acts is not None:
            group_config["input_activations"] = acts
        return group_config

    def _build_recipe(self) -> List[Any]:
        """Build the llmcompressor modifier recipe from the active config.

        Constructs an optional SmoothQuant modifier followed by the algorithm
        modifier (AWQ / GPTQ) or a plain QuantizationModifier (PTQ). The quant
        config lives inside ``config_groups``; ``targets`` and ``ignore`` sit on
        the modifier itself (NOT inside config_groups).

        Returns:
            List of llmcompressor Modifier instances for oneshot().
        """
        targets = self.config.scheme.targets
        ignore = self._merged_ignore()
        group_config = self._build_quant_config()
        config_groups = {"group_0": group_config}

        modifiers: List[Any] = []

        # SmoothQuant first (must precede any quantization modifier).
        sq = self.config.smoothquant
        if sq and sq.enabled:
            mappings = [(entry[0], entry[1]) for entry in sq.mappings]
            modifiers.append(
                SmoothQuantModifier(smoothing_strength=sq.strength, mappings=mappings)
            )
            logger.info("Enabled SmoothQuant (strength=%.2f, %d mappings)",
                        sq.strength, len(mappings))

        # Algorithm modifier: AWQ / GPTQ own config_groups; PTQ uses QuantizationModifier.
        if self.config.method == "awq":
            duo = self.config.awq.duo_scaling if self.config.awq else False
            modifiers.append(AWQModifier(
                targets=targets, ignore=ignore,
                config_groups=config_groups, duo_scaling=duo,
            ))
            logger.info("Using AWQ (duo_scaling=%s)", duo)
        elif self.config.method == "gptq":
            gptq = self.config.gptq
            modifiers.append(GPTQModifier(
                targets=targets, ignore=ignore, config_groups=config_groups,
                dampening_frac=gptq.dampening_frac if gptq else 0.01,
                block_size=gptq.block_size if gptq else 128,
            ))
            logger.info("Using GPTQ (dampening_frac=%.4f)",
                        gptq.dampening_frac if gptq else 0.01)
        else:  # ptq
            modifiers.append(QuantizationModifier(
                targets=targets, ignore=ignore, config_groups=config_groups,
            ))
            logger.info("Using plain PTQ (QuantizationModifier)")

        logger.info("Recipe scheme=%s weights=%s",
                    self.config.scheme.scheme, group_config["weights"])
        return modifiers

