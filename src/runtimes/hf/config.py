"""Runtime-side TurboQuant cache configuration.

Plain dataclass consumed by ``TurboQuantCache`` / ``VLMCrossCache`` in
``cache.py``. Populated by ``HFRuntime`` from the benchmark config's
``TurboQuantRuntimeConfig`` (see src/config/schemas.py) when TurboQuant is
enabled for a model entry.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Parameters for the TurboQuant KV cache.

    Attributes:
        ring_capacity: Most-recent tokens kept exact (bf16) in the ring buffer.
        key_bits:      Bit budget for compressed keys.
        value_bits:    Bit budget for compressed values.
        use_qjl:       If True, reconstruct keys with the QJL residual term. This
                       is an unbiased *inner-product* estimator and adds variance
                       to point-wise reconstruction — leave False (MSE-only) for
                       generation quality (low-bit keys suffer from outlier channels).
        use_compressed_store: If True, attention reads the decompressed overflow
                       store (TQ in the decode loop). If False, only the exact
                       ring is read (a sliding window; TQ measured for storage only).
        device:        Torch device string for the cache tensors.
        dtype:         Torch dtype name for the exact ring buffer.
    """
    ring_capacity: int = 256
    key_bits: int = 3
    value_bits: int = 2
    use_qjl: bool = False
    use_compressed_store: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
