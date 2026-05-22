# quantization/llmcompressor_quantizer.py
from __future__ import annotations
import logging
import torch
from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from src.config.schemas import QuantizeConfig
from .base import BaseQuantizer

logger = logging.getLogger(__name__)


class LLMCompressorQuantizer(BaseQuantizer):
    def __init__(self, config: QuantizeConfig):
        super().__init__(config)
        self._profile = None  # optional: use arch_profile from detect_profile

    def quantize(self) -> None:
        """Run llmcompressor oneshot with full configuration."""
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
        logger.info("Quantization finished and model saved to %s", self.config.output.output_dir)

    # -------------------------------------------------------------------------
    # Calibration data loading (multimodal)
    # -------------------------------------------------------------------------
    def _load_calibration_dataset(self) -> Dataset:
        cal = self.config.calibration
        split = getattr(cal, 'split', 'train')
        if "[:" not in split:
            split = f"{split}[:{cal.num_samples}]"

        try:
            ds = load_dataset(cal.dataset_name, split=split)
        except ValueError as e:
            from datasets import get_dataset_split_names
            splits = get_dataset_split_names(cal.dataset_name)
            raise ValueError(
                f"Split '{split}' not found. Available splits: {splits}. "
                f"Please set 'calibration.split' in your config."
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
            """For chat-format datasets like ultrachat_200k."""
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
            return {k: v for k, v in processed.items()}

        def preprocess_image_text(example):
            """
            For image-text pair datasets like LaTeX_OCR.
            Builds a single-turn VLM message:
            user:      [image] + instruction prompt
            assistant: ground-truth text (formula / caption / label)
            """
            from PIL import Image as PILImage

            image  = example[image_col]
            answer = str(example[text_col])

            # Ensure RGB PIL image — Qwen image processor rejects grayscale/palette/tiny.
            if not isinstance(image, PILImage.Image):
                image = PILImage.fromarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            w, h = image.size
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

            return {k: v for k, v in processed.items()}

        # ── Apply the right preprocessor ─────────────────────────────────────
        if dataset_format == 'chat':
            ds = ds.map(preprocess_chat,       batched=False, remove_columns=ds.column_names)
        else:
            ds = ds.map(preprocess_image_text, batched=False, remove_columns=ds.column_names)

        return ds

    def _data_collator(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of single sample into a tensor dictionary.
        Handles pixel_values for VLM.
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

    # -------------------------------------------------------------------------
    # Vision tower detection and ignore pattern merging
    # -------------------------------------------------------------------------
    def _is_vlm(self) -> bool:
        """Heuristic to detect if the loaded model is a vision‑language model."""
        # Check config for vision-related attributes
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
        """Return regex patterns for common vision tower modules."""
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
        """
        Merge user‑provided ignore patterns with automatic vision‑tower patterns.
        Ensures vision modules are never quantized.
        """
        user_ignore = self.config.scheme.ignore
        vision_ignore = self._get_vision_ignore_patterns()
        # Merge without duplicates
        merged = list(user_ignore)
        for pat in vision_ignore:
            if pat not in merged:
                merged.append(pat)
        if vision_ignore:
            logger.info("Auto‑ignoring vision modules: %s", vision_ignore)
        return merged

    # -------------------------------------------------------------------------
    # Recipe construction with full configurability
    # -------------------------------------------------------------------------
    def _build_recipe(self) -> List[Any]:
        scheme = self.config.scheme.scheme
        group_size = self.config.scheme.group_size
        symmetric = self.config.scheme.symmetric
        actorder = self.config.scheme.actorder
        targets = self.config.scheme.targets

        observer = getattr(self.config.scheme, "observer", "mse")
        supported_observers = {"mse", "minmax"}
        if observer not in supported_observers:
            logger.warning(
                "Observer '%s' not supported by llmcompressor, falling back to 'mse'", observer
            )
            observer = "mse"

        per_channel = getattr(self.config.scheme, "per_channel", False)
        dynamic_activations = getattr(self.config.scheme, "dynamic_activations", False)

        # Bit width
        if scheme.startswith("W4"):
            bits = 4
        elif scheme.startswith("W8"):
            bits = 8
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

        strategy = "channel" if per_channel else "group"

        # Build unified quant config — this goes INTO the algorithm modifier
        quant_config = {
            "targets": targets,
            "weights": {
                "num_bits": bits,
                "type": "int",
                "symmetric": symmetric,
                "group_size": group_size,
                "strategy": strategy,
                "dynamic": False,
                "actorder": actorder or None,
                "observer": observer,
            }
        }

        if "A8" in scheme:
            quant_config["activations"] = {
                "num_bits": 8,
                "type": "int",
                "symmetric": symmetric,
                "dynamic": dynamic_activations,
                "observer": observer,
            }
            if dynamic_activations:
                logger.info("Using dynamic activation quantization")

        modifiers = []

        # ── 1. SmoothQuant (must come first, before any quantization) ────────────
        sq = self.config.smoothquant
        if sq and sq.enabled:
            if "A8" not in scheme:
                raise ValueError(
                    "SmoothQuant requires activation quantization (scheme with A8)"
                )
            mappings = [(entry[0], entry[1]) for entry in sq.mappings]
            modifiers.append(SmoothQuantModifier(
                smoothing_strength=sq.strength,
                mappings=mappings,
            ))
            logger.info(
                "Enabled SmoothQuant with strength=%.2f on %d mapping groups",
                sq.strength, len(mappings)
            )

        # ── 2. Algorithm modifier — owns config_groups and ignore ────────────────
        #       NO separate QuantizationModifier when using GPTQ or AWQ
        if self.config.method == "awq":
            duo = self.config.awq.duo_scaling if self.config.awq else False
            modifiers.append(AWQModifier(
                duo_scaling=duo,
                ignore=self._merged_ignore(),
                config_groups={"group_0": quant_config},  # ← quant config lives here
            ))
            logger.info("Using AWQ with duo_scaling=%s", duo)

        elif self.config.method == "gptq":
            damp = self.config.gptq.dampening_frac if self.config.gptq else 0.01
            modifiers.append(GPTQModifier(
                dampening_frac=damp,
                # sequential_update removed in llmcompressor 0.6.0 — no longer a valid field
                ignore=self._merged_ignore(),
                config_groups={"group_0": quant_config},
            ))
            logger.info("Using GPTQ with dampening_frac=%.4f", damp)

        else:
            # PTQ only — QuantizationModifier is correct here since there's no algo modifier
            modifiers.append(QuantizationModifier(
                ignore=self._merged_ignore(),
                config_groups={"group_0": quant_config},
            ))
            logger.info("Using plain PTQ (no algorithm modifier)")

        logger.info(
            "Quantization config: bits=%d, group_size=%d, symmetric=%s, "
            "strategy=%s, observer=%s",
            bits, group_size, symmetric, strategy, observer,
        )
        return modifiers