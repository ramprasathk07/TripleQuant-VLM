import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Dict, Any, Type
import logging


# Configurable Parameters
# Model and Generation
MODEL_NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
IMAGE_PATH = "D:/personal/code_practice/TripleQuant-VLM/src/turboquant_v1/tests/sample.webp"
PROMPT = "What is shown in this image? Describe in detail."
MAX_NEW_TOKENS = 512

# TurboQuant Cache Parameters
RING_CAPACITY = 256
KEY_BITS = 3
VALUE_BITS = 2
USE_QJL = False
USE_COMPRESSED_STORE = True

# TurboQuant Core Logic (Compression Mixin)
class TurboQuantCompressionMixin:
    """Mixin that provides KV compression logic using your external library."""
    
    def _init_compression_engine(self, layer_idx: int, num_kv_heads: int, 
                                 head_dim: int, device: torch.device):
        # Import your turboquant components - adjust paths as needed
        try:
            from src.turboquant_v1 import ( 
                CompressedKVStore,
                KVCaptureEngine, 
                dequantize_V, 
                MSEQuantized
            )

        except ImportError as e:
            raise ImportError(
                "TurboQuant modules not found. Make sure your project root is in sys.path.\n"
                f"Original error: {e}"
            )
        
        self._CompressedKVStore = CompressedKVStore
        self._KVCaptureEngine = KVCaptureEngine
        self._dequantize_V = staticmethod(dequantize_V)
        self._MSEQuantized = MSEQuantized
        
        self.store = CompressedKVStore(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            key_bits=KEY_BITS,
            value_bits=VALUE_BITS,
            device=device,
            layer_idx=layer_idx,
            # use_qjl=USE_QJL
        )

        self.engine = self._KVCaptureEngine(
            store=self.store,
            ring_capacity=RING_CAPACITY,
            device=device,
            dtype=torch.bfloat16
        )

        self.num_tokens = 0
    
    def update_kv(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Update the cache with new key/value states."""
        # Convert from (1, H, T, D) to (T, H, D) format
        k = key_states[0].permute(1, 0, 2).contiguous()
        v = value_states[0].permute(1, 0, 2).contiguous()
        num_tokens = k.shape[0]
        self.engine.ingest_decode(k, v, num_tokens)
        self.num_tokens += num_tokens
    
    def get_full_kv(self, num_kv_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the full KV cache (compressed store + exact ring)."""
        recent = self.engine.ring.peek()
        if recent is not None:
            ring_k, ring_v = recent
        else:
            ring_k = torch.empty(0, num_kv_heads, self.engine.ring.head_dim, device=self.engine.ring.device)
            ring_v = torch.empty_like(ring_k)
        
        if USE_COMPRESSED_STORE and self.store.num_tokens > 0:
            flat = self.store.get_flat_cache()
            if USE_QJL:
                store_k = self.store.quantizer.dequantize(flat.prod_q)
            else:
                # MSE-only key reconstruction (better for point-wise quality)
                pq = flat.prod_q
                mse = self._MSEQuantized(indices=pq.mse_indices, norms=pq.norms, bits=pq.mse_bits)
                store_k = self.store.quantizer.mse_quantizer.dequantize(mse)
            store_v = self._dequantize_V(flat.value_q, group_size=self.store.value_group_size)
            
            store_k = store_k.permute(1, 0, 2).to(ring_k.dtype)
            store_v = store_v.permute(1, 0, 2).to(ring_v.dtype)
            return torch.cat([store_k, ring_k]), torch.cat([store_v, ring_v])
        
        return ring_k, ring_v
    
    def get_memory_bytes(self) -> int:
        """Get total memory usage of the cache."""
        return self.engine.ring.size * self.engine.ring.bytes_per_token + self.store.memory_bytes()



# General TurboDynamicCache (Works with any HF model)
class TurboDynamicCache(Cache, TurboQuantCompressionMixin):
    """A general-purpose HF-compatible KV cache with TurboQuant compression."""
    
    def __init__(self, model_config):
        super().__init__()
        
        # Read configuration from the model
        text_config = model_config.text_config   # or model.language_model.config
        self.num_layers = text_config.num_hidden_layers
        self.num_kv_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
        self.head_dim = text_config.hidden_size // text_config.num_attention_heads
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        
        # Create per-layer compression engines
        self.layer_engines: List[TurboQuantCompressionMixin] = []
        for i in range(self.num_layers):
            engine = TurboQuantCompressionMixin()
            engine._init_compression_engine(i, self.num_kv_heads, self.head_dim, self.device)
            self.layer_engines.append(engine)
        
        self._seq_len = 0
        self.is_compileable = False
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: int, cache_kwargs: Optional[Dict] = None):
        """Update the cache for a specific layer."""
        self.layer_engines[layer_idx].update_kv(key_states, value_states)
        if layer_idx == 0:
            self._seq_len += key_states.shape[2]  # seq_len dimension
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get the total sequence length in the cache."""
        return self._seq_len
    
    def get_max_length(self) -> Optional[int]:
        """Return the maximum cache length (None = unlimited)."""
        return None
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Handle beam search (not implemented for compression)."""
        raise NotImplementedError("Beam search is not supported with compressed KV cache.")
    
    def get_full_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the full KV cache (compressed store + ring) for a specific layer."""
        return self.layer_engines[layer_idx].get_full_kv(self.num_kv_heads)
    
    def get_memory_usage(self) -> int:
        """Get total memory usage across all layers."""
        return sum(engine.get_memory_bytes() for engine in self.layer_engines)



# Model-Agnostic Attention Patching
DECODER_ATTN_ORIG_FORWARDS = {}

def patch_attention_for_cache(model):
    """
    Generic attention patching that works for any HF model family.
    It finds all attention modules and replaces their forward method
    with a version that uses our TurboDynamicCache.
    """
    # Get the text transformer base class (works for LLM or VLM's LLM part)
    if hasattr(model, "language_model"):
        transformer = model.language_model.model
    elif hasattr(model, "model"):
        transformer = model.model
    else:
        transformer = model
    
    # Find the attention module class used by this model
    attn_module_class = None
    for module in transformer.modules():
        class_name = module.__class__.__name__
        if "Attention" in class_name and class_name not in DECODER_ATTN_ORIG_FORWARDS:
            attn_module_class = module.__class__
            DECODER_ATTN_ORIG_FORWARDS[attn_module_class] = module.forward
            break
    
    if attn_module_class is None:
        logging.warning("Could not find attention module to patch. Cache will not be used.")
        return
    
    def patched_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generic patched forward for any attention class."""
        
        # If our custom cache is not being used, fall back to original implementation
        if not isinstance(past_key_value, TurboDynamicCache):
            return DECODER_ATTN_ORIG_FORWARDS[type(self)](
                self, hidden_states, position_embeddings, attention_mask,
                past_key_value, cache_position, **kwargs
            )
        
        # Get dimensions from the module's config
        bsz, q_len, _ = hidden_states.size()
        n_heads = self.config.num_attention_heads
        n_kv_heads = getattr(self.config, "num_key_value_heads", n_heads)
        head_dim = getattr(self, "head_dim", self.config.hidden_size // n_heads)
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings if available
        if position_embeddings is not None:
            cos, sin = position_embeddings
            try:
                # Try to import the standard HF apply_rotary_pos_emb
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            except ImportError:
                # Fallback: assume the model has its own implementation
                pass
        
        # Update the cache with new tokens and retrieve full history
        past_key_value.update(key_states, value_states, self.layer_idx)
        full_k, full_v = past_key_value.get_full_kv(self.layer_idx)
        
        # Expand KV heads for GQA (if needed)
        if n_heads != n_kv_heads:
            n_rep = n_heads // n_kv_heads
            full_k = full_k.unsqueeze(0).repeat_interleave(n_rep, dim=2).permute(0, 2, 1, 3)
            full_v = full_v.unsqueeze(0).repeat_interleave(n_rep, dim=2).permute(0, 2, 1, 3)
        else:
            full_k = full_k.unsqueeze(0).permute(0, 2, 1, 3)
            full_v = full_v.unsqueeze(0).permute(0, 2, 1, 3)
        
        # Determine if causal mask is needed (only for prefill)
        is_causal = (q_len == full_k.shape[2]) and q_len > 1
        
        # Efficient attention using PyTorch's SDPA
        attn_output = F.scaled_dot_product_attention(
            query_states, full_k, full_v, attn_mask=attention_mask, 
            dropout_p=0.0, is_causal=is_causal,
        )
        
        # Final output projection
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    
    # Apply the patch to the attention class
    attn_module_class.forward = patched_attention_forward
    logging.info(f"Successfully patched attention for {attn_module_class.__name__}")



# Compression Reporting Helper
def compression_report(cache: TurboDynamicCache) -> dict:
    """Print and return compression statistics."""
    bf16_bytes = 2
    fp16_per_token = cache.num_kv_heads * cache.head_dim * bf16_bytes * 2
    n_layers = len(cache.layer_engines)
    seq_len = cache.get_seq_length()
    
    fp16_total = seq_len * fp16_per_token * n_layers
    tq_total = cache.get_memory_usage()
    
    compression_ratio = fp16_total / tq_total if tq_total > 0 else float("nan")
    
    print("\n" + "=" * 60)
    print("TurboQuant KV Cache Compression Report")
    print("=" * 60)
    print(f"  Model Layers:        {n_layers}")
    print(f"  Sequence Length:     {seq_len}")
    print(f"  KV Heads:            {cache.num_kv_heads}")
    print(f"  Head Dim:            {cache.head_dim}")
    print(f"  Ring Capacity:       {RING_CAPACITY}")
    print(f"  Key/Value Bits:      {KEY_BITS}/{VALUE_BITS}")
    print(f"  Use Compressed Store: {USE_COMPRESSED_STORE}")
    print("-" * 60)
    print(f"  FP16 Baseline:       {fp16_total / (1024**2):.2f} MB")
    print(f"  TurboQuant Total:    {tq_total / (1024**2):.2f} MB")
    print(f"  Overall Compression: {compression_ratio:.2f}x")
    print("=" * 60 + "\n")
    
    return {
        "seq_len": seq_len,
        "fp16_bytes": fp16_total,
        "tq_bytes": tq_total,
        "compression_ratio": compression_ratio,
    }



# Main Execution
if __name__ == "__main__":
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    
    # Patch the model's attention to use our custom cache
    patch_attention_for_cache(model)
    
    # Load and prepare image (skip if no image path provided)
    image = None
    if IMAGE_PATH and Path(IMAGE_PATH).exists():
        from PIL import Image
        image = Image.open(IMAGE_PATH).convert("RGB")
    
    # Prepare chat-style input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} if image else {},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    # Remove empty image entry if no image
    if not image:
        messages[0]["content"] = [{"type": "text", "text": PROMPT}]
    
    # Process inputs using the chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    
    # Initialize our custom compressed cache
    turbo_cache = TurboDynamicCache(model.config)
    
    # Generate with the custom cache
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            past_key_values=turbo_cache,
        )
    
    # Decode and print the generated text
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Text:\n" + "-" * 40)
    print(generated_text)
    print("-" * 40)
    
    # Print compression report
    compression_report(turbo_cache)