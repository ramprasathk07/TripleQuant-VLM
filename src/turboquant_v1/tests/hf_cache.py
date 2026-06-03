"""
Stage 3: Full HF cache replacement for Qwen2.5-3B using TurboQuant.
No compression yet – all tokens kept in the ring buffer for exact generation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb

# TurboQuant imports – adjust if your package structure differs
from src.turboquant_v1.store import CompressedKVStore
from src.turboquant_v1.capture import KVCaptureEngine

# -------------------------- Configuration --------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
RING_CAPACITY = 4096          # Keep all tokens uncompressed for now
KEY_BITS = 3                  # Will be used when compression is enabled
VALUE_BITS = 2
MAX_NEW_TOKENS = 50
PROMPT = "whats the capital of India"

# ---------------------- TurboDynamicCache --------------------------
class TurboDynamicCache(Cache):
    """HF-compatible cache backed by one KVCaptureEngine per layer."""
    def __init__(self, num_layers, num_kv_heads, head_dim, ring_capacity, device):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.ring_capacity = ring_capacity

        # One engine per layer – ring buffer + compressed store (store empty for now)
        self.engines = [
            KVCaptureEngine(
                store=CompressedKVStore(
                    head_dim=head_dim,
                    num_kv_heads=num_kv_heads,
                    key_bits=KEY_BITS,
                    value_bits=VALUE_BITS,
                    device=device,
                    layer_idx=i
                ),
                ring_capacity=ring_capacity,
                device=device,
                dtype=torch.bfloat16
            )
            for i in range(num_layers)
        ]
        self._seq_len = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        key_states: (1, num_kv_heads, new_len, head_dim)
        value_states: same shape
        """
        # Remove batch dim and permute to (T, Hkv, D)
        k = key_states[0].permute(1, 0, 2).contiguous()  # (new_len, Hkv, D)
        v = value_states[0].permute(1, 0, 2).contiguous()
        num_tokens = k.shape[0]

        # Ingest into the ring buffer (overflows are automatically compressed)
        self.engines[layer_idx].ingest_decode(k, v, num_tokens)

        if layer_idx == 0:
            self._seq_len += num_tokens

    def get_seq_length(self, layer_idx=0):
        return self._seq_len

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search not supported")

    def __getitem__(self, layer_idx):
        # HF generation code inspects shapes of past tensors; return dummies
        key = torch.empty(1, self.num_kv_heads, self._seq_len, self.head_dim, device=self.device)
        value = torch.empty(1, self.num_kv_heads, self._seq_len, self.head_dim, device=self.device)
        return (key, value)

    def to_legacy_cache(self):
        return self

    def get_max_length(self):
        return None

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        raise NotImplementedError

    def get_full_kv(self, layer_idx):
        """
        Returns (K, V) as (T_total, Hkv, D) for the given layer.
        Currently only from the ring buffer (store is empty).
        """
        engine = self.engines[layer_idx]
        recent = engine.ring.peek()
        if recent is not None:
            ring_k, ring_v = recent   # (T_recent, Hkv, D)
        else:
            ring_k = torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device)
            ring_v = torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device)

        # TODO: later prepend compressed tokens from engine.store.decompress(...)
        return ring_k, ring_v


# -------------------- Patched Attention ----------------------------
# Use global variables for model config (set after loading)
model_config = None

original_qwen2_attn_forward = Qwen2Attention.forward

def patched_qwen2_attn_forward(
    self,
    hidden_states,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    if isinstance(past_key_value, TurboDynamicCache) and use_cache:
        bsz, q_len, _ = hidden_states.size()
        n_heads = model_config.num_attention_heads
        n_kv_heads = model_config.num_key_value_heads
        head_dim = model_config.hidden_size // n_heads

        # Q projection + RoPE
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        # Use the model-level rotary embedding (avoid self.rotary_emb)
        cos, sin = model.model.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Append new K/V to cache (in‑place)
        past_key_value.update(key_states, value_states, self.layer_idx)

        # Get all past keys & values
        full_k, full_v = past_key_value.get_full_kv(self.layer_idx)   # (T_total, Hkv, D)

        # GQA expansion: repeat KV heads to match Q heads
        gqa_ratio = n_heads // n_kv_heads
        full_k = full_k.unsqueeze(0).repeat_interleave(gqa_ratio, dim=2)  # (1, T, H, D)
        full_k = full_k.permute(0, 2, 1, 3)                               # (1, H, T, D)
        full_v = full_v.unsqueeze(0).repeat_interleave(gqa_ratio, dim=2).permute(0, 2, 1, 3)

        # Causal scaled dot‑product attention
        attn_output = F.scaled_dot_product_attention(
            query_states, full_k, full_v, attn_mask=None, dropout_p=0.0, is_causal=True
        )   # (1, H, Tq, D)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None   # (output, None for attention weights)
    else:
        return original_qwen2_attn_forward(
            self, hidden_states, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, **kwargs,
        )

# Apply the monkey-patch
Qwen2Attention.forward = patched_qwen2_attn_forward

# ---------------------- Main Generation ----------------------------
if __name__ == "__main__":
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Make config globally accessible for the patched function
    model_config = model.config

    # Create the compressed cache
    turbo_cache = TurboDynamicCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        ring_capacity=RING_CAPACITY,
        device=model.device,
    )

    prompt = PROMPT
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy decoding
            use_cache=True,
            past_key_values=turbo_cache,   # our custom cache
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)