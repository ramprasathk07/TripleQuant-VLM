"""Generic attention monkeypatch that routes a decoder model through a TurboQuant cache.

Rather than hardcoding model classes, attention modules are detected by duck typing:
any ``*Attention`` module exposing q/k/v/o projections and a ``layer_idx``. This
covers Llama, Qwen2, Qwen3, Mistral, Gemma, Phi, and similar decoder-only models
without per-architecture imports.

The patch only activates when ``past_key_value`` is one of our caches; for any other
cache it defers to the original forward, so unrelated models keep working.
"""
import torch.nn.functional as F

from .cache import TurboQuantCache, VLMCrossCache

# Maps an attention class -> its original unbound forward, so we can both restore it
# and fall back to it for non-TurboQuant caches.
_original_forwards = {}

_PROJ_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _rope(query, key, position_embeddings):
    """Apply rotary embeddings using the (cos, sin) the model already computed.

    The RoPE formula is identical across Llama/Qwen/Mistral/Gemma, so we reuse the
    Llama implementation rather than importing each model's copy.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    cos, sin = position_embeddings
    return apply_rotary_pos_emb(query, key, cos, sin)


def _generic_attn_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    # Only intercept our caches; everything else uses the model's own attention.
    if not isinstance(past_key_value, (TurboQuantCache, VLMCrossCache)):
        return _original_forwards[type(self)](
            self, hidden_states, position_embeddings, attention_mask,
            past_key_value, cache_position, **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()
    n_heads = self.config.num_attention_heads
    n_kv_heads = getattr(self.config, "num_key_value_heads", n_heads)
    head_dim = getattr(self, "head_dim", self.config.hidden_size // n_heads)

    query = self.q_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    key = self.k_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    value = self.v_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    # QK-norm (Qwen3, Gemma2, …): RMSNorm over head_dim applied to q/k before RoPE.
    # No-op for models without it (Llama/Qwen2/Mistral).
    if getattr(self, "q_norm", None) is not None:
        query = self.q_norm(query)
    if getattr(self, "k_norm", None) is not None:
        key = self.k_norm(key)

    if position_embeddings is not None:
        query, key = _rope(query, key, position_embeddings)

    past_key_value.update(key, value, self.layer_idx)
    full_k, full_v = past_key_value.get_full_kv(self.layer_idx)   # (T_total, Hkv, D)

    gqa_ratio = n_heads // n_kv_heads
    full_k = full_k.unsqueeze(0).repeat_interleave(gqa_ratio, dim=2).permute(0, 2, 1, 3)
    full_v = full_v.unsqueeze(0).repeat_interleave(gqa_ratio, dim=2).permute(0, 2, 1, 3)

    # Causal only at prefill (q_len == kv_len > 1); a single decode query (q_len==1)
    # must attend the full history, so is_causal=False there.
    is_causal = (q_len == full_k.shape[2]) and q_len > 1
    attn_out = F.scaled_dot_product_attention(
        query, full_k, full_v, attn_mask=None, dropout_p=0.0, is_causal=is_causal,
    )
    attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1)
    attn_out = self.o_proj(attn_out)
    return attn_out, None


def _is_attention_module(module) -> bool:
    """Duck-type an attention module: q/k/v/o projections + a layer index."""
    return (
        type(module).__name__.endswith("Attention")
        and all(hasattr(module, a) for a in _PROJ_ATTRS)
        and hasattr(module, "layer_idx")
    )


def patch_model_attention(model) -> int:
    """Replace the forward of every decoder attention module. Returns count patched."""
    patched = 0
    for module in model.modules():
        if not _is_attention_module(module):
            continue
        cls = type(module)
        if cls not in _original_forwards:
            _original_forwards[cls] = cls.forward
        module.forward = _generic_attn_forward.__get__(module, cls)
        patched += 1
    return patched
