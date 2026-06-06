import torch
from transformers.cache_utils import Cache
from typing import Optional, Tuple, List

# Import your turboquant components (adjust paths)
from src.turboquant_v1 import (CompressedKVStore,
                               KVCaptureEngine,
                               dequantize_V,
                               MSEQuantized
                            )
from .config import CacheConfig

class TurboQuantCache(Cache):
    """General KV cache with ring buffer + compressed overflow."""
    def __init__(self, model_config, cfg: CacheConfig):
        super().__init__()
        self.cfg = cfg
        self.num_layers = model_config.num_hidden_layers
        self.num_kv_heads = getattr(model_config, "num_key_value_heads",
                                    model_config.num_attention_heads)
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        
        self.engines: List[KVCaptureEngine] = []
        for i in range(self.num_layers):
            store = CompressedKVStore(
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                key_bits=cfg.key_bits,
                value_bits=cfg.value_bits,
                device=self.device,
                layer_idx=i,
            )
            engine = KVCaptureEngine(
                store=store,
                ring_capacity=cfg.ring_capacity,
                device=self.device,
                dtype=self.dtype
            )
            self.engines.append(engine)
        self._seq_len = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # (batch=1, num_heads, seq_len, head_dim) -> (seq_len, num_heads, head_dim)
        k = key_states[0].permute(1, 0, 2).contiguous()
        v = value_states[0].permute(1, 0, 2).contiguous()
        self.engines[layer_idx].ingest_decode(k, v, k.shape[0])
        if layer_idx == 0:
            self._seq_len += k.shape[0]

    def get_full_kv(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (K, V) as (T_total, num_heads, head_dim) in chronological order."""
        engine = self.engines[layer_idx]
        ring_k, ring_v = engine.ring.peek() or (None, None)
        if ring_k is None:
            ring_k = torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device)
            ring_v = torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device)
        
        if self.cfg.use_compressed_store and engine.store.num_tokens > 0:
            flat = engine.store.get_flat_cache()
            if self.cfg.use_qjl:
                store_k = engine.store.quantizer.dequantize(flat.prod_q)
            else:
                # MSE-only reconstruction
                mse = MSEQuantized(indices=flat.prod_q.mse_indices,
                                   norms=flat.prod_q.norms,
                                   bits=flat.prod_q.mse_bits)
                store_k = engine.store.quantizer.mse_quantizer.dequantize(mse)
            store_v = dequantize_V(flat.value_q, group_size=engine.store.value_group_size)
            store_k = store_k.permute(1, 0, 2).to(ring_k.dtype)
            store_v = store_v.permute(1, 0, 2).to(ring_v.dtype)
            return torch.cat([store_k, ring_k]), torch.cat([store_v, ring_v])
        return ring_k, ring_v

    # HF Cache protocol methods
    def get_seq_length(self, layer_idx=0) -> int:
        return self._seq_len
    
    def get_max_length(self) -> Optional[int]:
        return None
    
    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search not supported")
    
    def to_legacy_cache(self):
        return self


class VLMCrossCache(Cache):
    """Separate cache for cross‑attention (visual tokens) in VLMs."""
    def __init__(self, num_layers, num_heads, head_dim, cfg: CacheConfig):
        super().__init__()
        self.engines = [
            KVCaptureEngine(
                store=CompressedKVStore(
                    head_dim=head_dim, num_kv_heads=num_heads,
                    key_bits=cfg.key_bits, value_bits=cfg.value_bits,
                    device=torch.device(cfg.device), layer_idx=i,
                ),
                ring_capacity=cfg.ring_capacity,
                device=torch.device(cfg.device),
                dtype=getattr(torch, cfg.dtype)
            )
            for i in range(num_layers)
        ]
        self._seq_len = 0
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states shape: (batch, num_heads, seq_len, head_dim)
        k = key_states[0].permute(1, 0, 2).contiguous()
        v = value_states[0].permute(1, 0, 2).contiguous()
        self.engines[layer_idx].ingest_decode(k, v, k.shape[0])
        if layer_idx == 0:
            self._seq_len += k.shape[0]
    
    def get_full_kv(self, layer_idx):
        ring_k, ring_v = self.engines[layer_idx].ring.peek() or (None, None)
        if ring_k is None:
            ring_k = torch.empty(0, self.engines[layer_idx].store.num_kv_heads,
                                 self.engines[layer_idx].store.head_dim,
                                 device=self.engines[layer_idx].store.device)
            ring_v = torch.empty_like(ring_k)
        # For simplicity, we do not read compressed store in cross-attention (visual tokens are often static)
        return ring_k, ring_v
    
    def get_seq_length(self, layer_idx=0):
        return self._seq_len
    
    def reorder_cache(self, beam_idx):
        raise NotImplementedError