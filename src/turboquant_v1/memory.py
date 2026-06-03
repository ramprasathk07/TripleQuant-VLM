from __future__ import annotations

import torch
from typing import Optional, NamedTuple

from turboquant_v1.quantize import TurboQuant, ProdQuantized
from turboquant_v1.kv_cache import quantize_V, ValueQuantized


class FlatCache(NamedTuple):
    """Flattened view of compressed KV for fast read access."""
    prod_q: ProdQuantized       # (num_kv_heads, total_tokens, ...)
    value_q: ValueQuantized     # (num_kv_heads, total_tokens, ...)
    num_tokens: int


class CompressedKVStore:
    """Chunked compressed KV store with lazy flattening.

    Keys are quantized via TurboQuantProd (unbiased inner-product estimator).
    Values use symmetric group quantization.
    Chunks are kept in lists until a flat view is requested.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        device: torch.device = None,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = min(value_group_size, head_dim)
        self.device = device or torch.device("cuda")
        self.layer_idx = layer_idx

        self.quantizer = TurboQuant(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        self._key_chunks: list[ProdQuantized] = []
        self._value_chunks: list[ValueQuantized] = []
        self._chunk_lengths: list[int] = []

        self._flat: Optional[FlatCache] = None

    @property
    def num_tokens(self) -> int:
        return sum(self._chunk_lengths)

    @property
    def num_chunks(self) -> int:
        return len(self._chunk_lengths)

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        """Quantize and store a chunk of KV pairs.

        key/value: (chunk_len, num_kv_heads, head_dim)
        """
        chunk_len = key.shape[0]

        # Reshape to (1, num_kv_heads, chunk_len, head_dim) for quantizer
        k = key.transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
        v = value.transpose(0, 1).unsqueeze(0)

        key_q = self.quantizer.quantize(k)
        val_q = quantize_V(v, bits=self.value_bits, group_size=self.value_group_size)

        self._key_chunks.append(key_q)
        self._value_chunks.append(val_q)
        self._chunk_lengths.append(chunk_len)
        self._flat = None  # invalidate