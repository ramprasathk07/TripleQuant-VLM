"""
Stage 3: Full HF cache replacement for Qwen2.5-3B using TurboQuant.
No compression yet – all tokens kept in the ring buffer for exact generation.
"""

import sys
from pathlib import Path

# Allow running this file directly (`python src/turboquant_v1/tests/hf_cache.py`):
# put the project root on sys.path so `import src...` resolves. Running a script
# directly only adds the script's own dir to sys.path, not the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb

# TurboQuant imports – adjust if your package structure differs
from src.turboquant_v1.store import CompressedKVStore
from src.turboquant_v1.capture import KVCaptureEngine
from src.turboquant_v1.kv_cache import dequantize_V

# -------------------------- Configuration --------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
RING_CAPACITY = 256          # Keep all tokens uncompressed for now
# Bit budget for the compressed store. The real fidelity lever here is USE_QJL,
# not the bit count: with +QJL, K3 keys are garbage; with MSE-only, K3 keys hit
# cos ~0.94 (see test_quant_quality.py). Measured on Qwen2.5-3B:
#   K3/V2, MSE-only -> 5.12x segment, COHERENT  <- default
#   K3/V2, +QJL     -> 5.12x segment, garbage
#   K8/V4           -> 2.25x segment, clean (works regardless of QJL)
KEY_BITS = 3
VALUE_BITS = 2

# The QJL residual stage is an unbiased *inner-product* estimator; reused for
# point-wise key reconstruction it injects variance and RAISES error at every bit
# width (K3 cos 0.94 -> 0.92). Skip it for the dequant/score path — MSE-only keys
# are closer to the originals. See src/turboquant_v1/tests/test_quant_quality.py.
USE_QJL = False

# When True, get_full_kv decompresses the store and prepends it, so attention
# actually runs over TurboQuant-reconstructed (lossy) KV — TQ is in the decode
# loop. When False, attention reads the exact bf16 ring only (a sliding window
# numerically identical to a plain bf16 cache; TQ measured for storage only).
USE_COMPRESSED_STORE = True

MAX_NEW_TOKENS = 2048
# Force at least this many new tokens so the ring (capacity 128) overflows and the
# compressed store actually fills — otherwise a short answer never triggers
# compression and the report shows 1.00x. Output past the real answer degenerates
# because get_full_kv currently reads the ring only (store not yet read back).
# MIN_NEW_TOKENS = 512
PROMPT = "explain elaborately on whats llm and AI"

# ---------------------- TurboDynamicCache --------------------------
class TurboDynamicCache(Cache):
    """HF-compatible cache backed by one KVCaptureEngine per layer."""

    # transformers >=4.55 reads these off the cache during generation. This cache
    # has no base `layers` list and is not torch.compile-able, so pin them
    # directly (the base `is_compileable` property iterates self.layers).
    is_compileable = False

    def __init__(self, num_layers, num_kv_heads, head_dim, ring_capacity, device):
        # transformers >=4.55 changed Cache.__init__ to require `layer_classes`
        # (layered-cache refactor). This cache manages its own per-layer engines
        # and overrides every access point HF generation uses, so it does not
        # rely on the base init — skip it when the new signature rejects no-args.
        try:
            super().__init__()
        except TypeError:
            pass
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

    def get_mask_sizes(self, cache_position, layer_idx=0):
        # (kv_length, kv_offset) for causal-mask construction. transformers >=4.55
        # delegates to self.layers[idx]; this cache has none, so compute directly.
        # Called before the decoder layers run, so _seq_len is the PAST length and
        # the current query tokens (cache_position) are added on top.
        query_length = cache_position.shape[0]
        return self._seq_len + query_length, 0

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
        Returns (K, V) as (T_total, Hkv, D) for the given layer, in chronological
        order: TurboQuant-reconstructed store tokens (oldest) followed by the exact
        bf16 ring tokens (most recent).

        With USE_COMPRESSED_STORE the store is decompressed and prepended, so the
        model attends over lossy TQ-reconstructed KV — TQ is genuinely in the decode
        loop. Otherwise only the ring is returned (exact bf16 sliding window).
        """
        engine = self.engines[layer_idx]
        recent = engine.ring.peek()
        if recent is not None:
            ring_k, ring_v = recent   # (T_recent, Hkv, D)
        else:
            ring_k = torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device)
            ring_v = torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device)

        if USE_COMPRESSED_STORE and engine.store.num_tokens > 0:
            flat = engine.store.get_flat_cache()
            # Keys: TurboQuant inverse. Values: group-dequant. Both -> (Hkv, T, D).
            if USE_QJL:
                store_k = engine.store.quantizer.dequantize(flat.prod_q)
            else:
                # MSE-only key reconstruction: the QJL residual stage is an
                # unbiased *inner-product* estimator and injects variance into
                # point-wise reconstruction (measured: it raises rel-err / lowers
                # cos at every bit width). Skipping it keeps keys closer.
                from src.turboquant_v1.quantize import MSEQuantized
                pq = flat.prod_q
                mse = MSEQuantized(indices=pq.mse_indices, norms=pq.norms, bits=pq.mse_bits)
                store_k = engine.store.quantizer.mse_quantizer.dequantize(mse)
            store_v = dequantize_V(flat.value_q, group_size=engine.store.value_group_size)
            # -> (T, Hkv, D) to match the ring layout, in the cache's dtype.
            store_k = store_k.permute(1, 0, 2).to(ring_k.dtype)
            store_v = store_v.permute(1, 0, 2).to(ring_v.dtype)
            return (torch.cat([store_k, ring_k], dim=0),
                    torch.cat([store_v, ring_v], dim=0))

        return ring_k, ring_v


# -------------------- Patched Attention ----------------------------
# Use global variables for model config (set after loading)
model_config = None

original_qwen2_attn_forward = Qwen2Attention.forward

def patched_qwen2_attn_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    # transformers >=4.55 attention signature is
    #   (hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
    # and the decoder passes everything by keyword, so gate on the cache type
    # (not a `use_cache` flag, which is no longer forwarded here).
    if not isinstance(past_key_value, TurboDynamicCache):
        return original_qwen2_attn_forward(
            self, hidden_states, position_embeddings, attention_mask,
            past_key_value=past_key_value, cache_position=cache_position, **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()
    cfg = getattr(self, "config", model_config)
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(self, "head_dim", cfg.hidden_size // n_heads)

    query_states = self.q_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    # Use the precomputed (cos, sin) the model already built for this step; do NOT
    # recompute from position_ids (4.55 no longer passes position_ids here).
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Append post-RoPE K/V to the cache (in-place), then read the full history.
    past_key_value.update(key_states, value_states, self.layer_idx)
    full_k, full_v = past_key_value.get_full_kv(self.layer_idx)   # (T_total, Hkv, D)

    # GQA: expand KV heads to match Q heads. (T,Hkv,D) -> (1,Hq,T,D).
    gqa_ratio = n_heads // n_kv_heads
    full_k = full_k.unsqueeze(0).repeat_interleave(gqa_ratio, dim=2).permute(0, 2, 1, 3)
    full_v = full_v.unsqueeze(0).repeat_interleave(gqa_ratio, dim=2).permute(0, 2, 1, 3)

    # is_causal only when query and key lengths match (prefill). For a decode step
    # (q_len == 1, kv_len == T) is_causal=True would let the single query see only
    # key 0 — pass False so it attends the full history.
    is_causal = q_len == full_k.shape[2] and q_len > 1
    attn_output = F.scaled_dot_product_attention(
        query_states, full_k, full_v, attn_mask=None, dropout_p=0.0, is_causal=is_causal,
    )   # (1, Hq, q_len, D)

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None   # (output, attn_weights)

# Apply the monkey-patch
Qwen2Attention.forward = patched_qwen2_attn_forward


# -------------------- Compression reporting ------------------------
def compression_report(cache: "TurboDynamicCache") -> dict:
    """Compare TurboQuant KV storage against the FP16 KV you'd keep without it.

    Per layer the cache holds two segments:
      - ring buffer: the most recent <= ring_capacity tokens, exact (bf16)
      - compressed store: the overflow tokens, quantized to KEY_BITS / VALUE_BITS

    "Without TQ" is the dense FP16 KV cache: every token kept at bf16 for both K
    and V. The ratio is how many times smaller the TurboQuant cache is.

    NOTE: today get_full_kv() reads the ring only, so the store is write-only and
    does not feed attention (sliding window of ring_capacity). These numbers
    measure storage; reconstructing from the store is the next step.
    """
    bf16_bytes = 2
    h_kv, d = cache.num_kv_heads, cache.head_dim
    # K and V, per token, per layer, at bf16.
    fp16_per_token = h_kv * d * bf16_bytes * 2

    ring_tokens = store_tokens = 0
    tq_bytes = 0
    store_only_bytes = 0
    for eng in cache.engines:
        r = eng.ring.size
        s = eng.store.num_tokens
        ring_tokens += r
        store_tokens += s
        sb = eng.store.memory_bytes()        # store's own estimate of packed size
        store_only_bytes += sb
        tq_bytes += r * fp16_per_token + sb  # ring stays bf16; store is packed

    seq_len = cache.get_seq_length()
    n_layers = len(cache.engines)
    fp16_bytes = seq_len * fp16_per_token * n_layers      # dense FP16 cache (no TQ)
    store_fp16_equiv = store_tokens * fp16_per_token       # FP16 cost of the compressed part

    mb = 1024 * 1024
    overall = fp16_bytes / tq_bytes if tq_bytes else float("nan")
    store_only = store_fp16_equiv / store_only_bytes if store_only_bytes else float("nan")

    print("\n" + "-" * 58)
    print("KV cache compression report")
    print("-" * 58)
    print(f"  layers                 : {n_layers}")
    print(f"  sequence length        : {seq_len} tokens")
    print(f"  ring_capacity          : {cache.ring_capacity}  (KEY_BITS={KEY_BITS}, VALUE_BITS={VALUE_BITS})")
    print(f"  per-layer split        : {ring_tokens // n_layers} exact (bf16) + "
          f"{store_tokens // n_layers} compressed")
    print(f"  FP16 KV  (without TQ)  : {fp16_bytes / mb:8.2f} MB")
    print(f"  TurboQuant KV          : {tq_bytes / mb:8.2f} MB  "
          f"(ring bf16 + store packed)")
    print(f"  overall compression    : {overall:6.2f}x")
    print(f"  compressed-segment only: {store_only:6.2f}x  "
          f"({store_fp16_equiv / mb:.2f} MB -> {store_only_bytes / mb:.2f} MB)")
    print("-" * 58)
    return {
        "seq_len": seq_len,
        "fp16_bytes": fp16_bytes,
        "tq_bytes": tq_bytes,
        "overall_ratio": overall,
        "store_only_ratio": store_only,
    }

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
            # min_new_tokens=MIN_NEW_TOKENS,   # force ring overflow so the store fills
            do_sample=False,          # greedy decoding
            use_cache=True,
            past_key_values=turbo_cache,   # our custom cache
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)

    compression_report(turbo_cache)