"""torchao quantization backend.

A third quantization backend (alongside llm_compressor and modelopt) built on
PyTorch's `torchao`. The model is loaded normally (BaseQuantizer.load_model) and
quantized in place with `torchao.quantize_`, then exported. We apply torchao
directly rather than through the transformers `TorchAoConfig` integration because
that integration is sensitive to the transformers/torchao version pairing.

Weight-only int8/int4 and dynamic-activation int8 work on Ampere+. int4/fp8 need a
recent torch / Ada+ GPU; on older setups quantize_ raises and quantize.py reports it.

Scheme -> torchao config:
    W8A16        -> Int8WeightOnlyConfig
    W4A16(_ASYM) -> Int4WeightOnlyConfig(group_size)
    W8A8         -> Int8DynamicActivationInt8WeightConfig
    FP8          -> Float8WeightOnlyConfig
    FP8_DYNAMIC  -> Float8DynamicActivationFloat8WeightConfig

Note: the export uses ``safe_serialization=False`` (torchao tensor subclasses are
not safetensors-serializable); reload the checkpoint with ``from_pretrained`` in an
environment that has torchao installed.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseQuantizer

logger = logging.getLogger(__name__)


class TorchAOQuantizer(BaseQuantizer):
    """Quantize a model with torchao (weight-only int8/int4, dynamic int8, fp8)."""

    def _torchao_config(self):
        from torchao.quantization import (
            Int8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
        )
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
            Int4PackingFormat,
        )
        gs = self.config.scheme.group_size
        # TILE_PACKED_TO_4D: the only int4 packing format that runs on this
        # torchao 0.17 / sm_86 stack — PLAIN/PRESHUFFLED need the uninstallable
        # 'mslk' kernel package (see docs/qwen3_1_7b_leaderboard.md).
        _int4 = lambda: Int4WeightOnlyConfig(
            group_size=gs, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
        builders = {
            "W8A16":      Int8WeightOnlyConfig,
            "W4A16":      _int4,
            "W4A16_ASYM": _int4,
            "W8A8":       Int8DynamicActivationInt8WeightConfig,
        }
        try:  # fp8 needs Ada/Hopper + a torchao build that ships it
            from torchao.quantization import (
                Float8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig,
            )
            builders["FP8"] = Float8WeightOnlyConfig
            builders["FP8_DYNAMIC"] = Float8DynamicActivationFloat8WeightConfig
        except Exception:
            pass

        scheme = self.config.scheme.scheme
        if scheme not in builders:
            raise ValueError(
                f"torchao backend does not support scheme '{scheme}'. "
                f"Choose one of: {sorted(builders)}"
            )
        return builders[scheme]()

    def quantize(self) -> None:
        """Apply torchao quantization in place, then export the checkpoint."""
        from torchao.quantization import quantize_
        cfg = self._torchao_config()
        logger.info("Applying torchao %s (%s) ...",
                    self.config.scheme.scheme, type(cfg).__name__)
        quantize_(self.model, cfg)
        self.save(str(self.config.output.output_dir))

    def save(self, output_dir: str, weights: bool = True) -> None:
        """Save the torchao model + tokenizer/processor.

        Overrides BaseQuantizer.save: torchao tensor subclasses need
        ``safe_serialization=False`` and don't take the ``save_compressed`` kwarg.
        """
        name = self.model_id.split("/")[-1]
        subfolder = f"{name}-torchao-{self.config.method}-{self.config.scheme.scheme}"
        final_dir = Path(output_dir) / subfolder
        logger.info("Saving torchao model to: %s", final_dir)

        if weights:
            self.model.to("cpu")
            if hasattr(self.model, "hf_device_map"):
                del self.model.hf_device_map
            self.model.save_pretrained(str(final_dir), safe_serialization=False)
        if self.processor is not None:
            self.processor.save_pretrained(str(final_dir))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(final_dir))
        logger.info("torchao checkpoint saved -> %s", final_dir)
