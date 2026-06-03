import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from src.turboquant_v1.store import CompressedKVStore
from src.turboquant_v1.capture import KVCaptureEngine
from src.turboquant_v1.score import compute_hybrid_attention

model_name = "Qwen/Qwen2.5-3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

config = model.config
num_heads = config.num_attention_heads
num_key_value_heads = config.num_key_value_heads
head_dim = config.hidden_size // num_heads

rope = model.model.rotary_emb 

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

attn_module = model.model.layers[0].self_attn
original_forward = attn_module.forward

captured = {}

def capture_qkv(self, hidden_states, position_ids=None, **kwargs):
    bsz, sq_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape for multi‑head (same logic as Qwen2Attention)
    query_states = query_states.view(bsz, sq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, sq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, sq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply RoPE (uses Qwen’s rotary embeddings)
    cos, sin = rope(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Capture the final Q, K, V
    captured["q"] = query_states.detach()
    captured["k"] = key_states.detach()
    captured["v"] = value_states.detach()

    del query_states, key_states, value_states
    # Let the original forward recompute everything identically
    return original_forward(hidden_states, position_ids=position_ids, **kwargs)

attn_module.forward = capture_qkv.__get__(attn_module)

text = "Hello, world! " * 512 
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    _ = model(**inputs)

attn_module.forward = original_forward

Q = captured["q"]
K = captured["k"]
V = captured["v"]

n_groups = num_heads // num_key_value_heads   # 16 // 2 = 8
K_expanded = K.repeat_interleave(n_groups, dim=1)   # [1, 16, 4, 128]
V_expanded = V.repeat_interleave(n_groups, dim=1)   # for cache

scores_fp = torch.matmul(Q, K_expanded.transpose(-2, -1)) 

scores_fp = torch.matmul(
    Q.float(),
    K_expanded.float().transpose(-2, -1),
)

weights_fp = torch.softmax(
    scores_fp / (head_dim ** 0.5),
    dim=-1,
)

output_fp = torch.matmul(
    weights_fp,
    V_expanded.float(),
)

output_fp = output_fp[0].permute(1, 0, 2).contiguous()

# tq
store = CompressedKVStore(
    head_dim=head_dim,
    num_kv_heads=num_key_value_heads,
    key_bits=3,
    value_bits=2,
    device=K.device,
)

print("Store key_bits:", store.key_bits)
print("Store value_bits:", store.value_bits)

capture = KVCaptureEngine(
    store,
    ring_capacity=128,
)

Q = Q[0]
K = K[0]
V = V[0]

K_tq = K.permute(1,0,2).contiguous()
V_tq = V.permute(1,0,2).contiguous()

store.append_chunk(
    K_tq,
    V_tq,
)

Q_tq = Q.permute(1,0,2).contiguous()

output_tq = compute_hybrid_attention(
    query=Q_tq,
    store=store,
    recent_k=None,
    recent_v=None,
    num_query_heads=num_heads,
)

print(output_fp.shape)
print(output_tq.shape)

mse = ((output_fp.float() - output_tq.float())**2).mean()
mae = (output_fp.float() - output_tq.float()).abs().mean()

print("MSE:", mse.item())
print("MAE:", mae.item())

fp16_bytes = K.numel() * 2 + V.numel() * 2
tq_bits = K.numel() *  store.key_bits + V.numel() * store.value_bits
tq_bytes = tq_bits / 8

print(f"FP16 KV cache: {fp16_bytes:.2f} bytes")
print(f"TurboQuant KV cache: {tq_bytes:.2f} bytes")
print(f"Memory reduction: {(1 - tq_bytes/fp16_bytes)*100:.1f}%")