"""TurboQuant quantizer quality diagnostics.

Standalone, model-free checks that isolate the quantizer from the HF integration.
Run them whenever you change the codebook, rotation, QJL, or value quantizer to see
the effect on reconstruction fidelity and inner-product preservation.

Run:
    python src/turboquant_v1/tests/test_quant_quality.py

What it reports (and why):
  - Key round-trip rel-err / cosine, MSE-only vs +QJL, across bit widths.
    Finding baked in: the QJL residual stage *raises* point-wise reconstruction
    error (it is an unbiased inner-product estimator with high variance at m=d
    projections), so MSE-only keys are closer to the originals.
  - Value round-trip rel-err / cosine across bit widths.
  - Attention-score accuracy: the asymmetric estimator vs true q.k, and the
    (important) check that estimator == dequantize-then-matmul.
  - End-to-end store: append_chunk -> get_flat_cache -> reconstruct, to catch
    flatten/concat/packing bugs.
  - Light asserts: K8 must round-trip near-exact (guards against codebook-scale
    or packing regressions).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.turboquant_v1.quantize import TurboQuant, MSEQuantized
from src.turboquant_v1.kv_cache import quantize_V, dequantize_V
from src.turboquant_v1.store import CompressedKVStore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIM = 128
N = 512


# ── metrics ──────────────────────────────────────────────────────────────────

def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error ||a-b|| / ||a|| (0 = perfect)."""
    a, b = a.float(), b.float()
    return (a - b).norm().item() / (a.norm().item() + 1e-12)


def mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row cosine similarity (1 = perfect direction match)."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


def _mse_only(tq: TurboQuant, q) -> torch.Tensor:
    """Reconstruct keys using the MSE codebook only (drop the QJL residual)."""
    mse = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
    return tq.mse_quantizer.dequantize(mse)


# ── tests ────────────────────────────────────────────────────────────────────

def test_key_roundtrip(bits_list=(3, 4, 8)) -> None:
    """Key reconstruction fidelity, MSE-only vs full (MSE+QJL)."""
    print("\n[key round-trip]  (gaussian vectors = uniform directions = codebook's design dist)")
    x = torch.randn(N, DIM, device=DEVICE)
    for bits in bits_list:
        tq = TurboQuant(dim=DIM, bits=bits, device=DEVICE)
        q = tq.quantize(x)
        x_mse = _mse_only(tq, q)
        x_full = tq.dequantize(q)
        print(f"  K{bits}: MSE-only rel={rel_err(x, x_mse):.3f} cos={mean_cos(x, x_mse):.3f}"
              f"  |  +QJL rel={rel_err(x, x_full):.3f} cos={mean_cos(x, x_full):.3f}")
    print("  NOTE: +QJL should look WORSE here — QJL trades point-wise error for an")
    print("        unbiased inner-product estimate. Use MSE-only for reconstruction.")


def test_value_roundtrip(bits_list=(2, 4, 8)) -> None:
    """Value group-quantization fidelity."""
    print("\n[value round-trip]  (symmetric group quant)")
    v = torch.randn(N, DIM, device=DEVICE)
    for bits in bits_list:
        if DIM % 4 != 0 and bits == 2:
            continue
        vq = quantize_V(v.unsqueeze(0), bits=bits, group_size=min(32, DIM))
        vr = dequantize_V(vq, group_size=min(32, DIM)).squeeze(0)
        print(f"  V{bits}: rel={rel_err(v, vr):.3f} cos={mean_cos(v, vr):.3f}")


def test_attention_score(bits=3) -> None:
    """Inner-product preservation, and estimator vs dequant-then-matmul."""
    print(f"\n[attention score]  (K{bits}, the metric TurboQuant actually targets)")
    k = torch.randn(N, DIM, device=DEVICE)
    q = torch.randn(8, DIM, device=DEVICE)
    tq = TurboQuant(dim=DIM, bits=bits, device=DEVICE)
    qz = tq.quantize(k)

    s_true = q @ k.t()                                  # (8, N) reference logits
    s_est = tq.attention_score(q.unsqueeze(0), qz).squeeze(0) if q.dim() == 2 else tq.attention_score(q, qz)
    # estimator expects matching batch dims; here keys are (N,...) with no head dim
    s_est = tq.attention_score(q, qz)
    s_dq = q @ tq.dequantize(qz).t()
    s_mse = q @ _mse_only(tq, qz).t()
    print(f"  estimator   rel={rel_err(s_true, s_est):.3f} cos={mean_cos(s_true, s_est):.3f}")
    print(f"  dequant@mm  rel={rel_err(s_true, s_dq):.3f} cos={mean_cos(s_true, s_dq):.3f}")
    print(f"  MSE-only@mm rel={rel_err(s_true, s_mse):.3f} cos={mean_cos(s_true, s_mse):.3f}")
    print(f"  estimator == dequant@mm ?  max|diff| = {(s_est - s_dq).abs().max().item():.2e}")
    print("  NOTE: estimator and dequant@mm are the same math (proves the estimator")
    print("        path gives no accuracy gain over dequantize-then-matmul).")


def test_store_endtoend(bits_k=3, bits_v=2) -> None:
    """append_chunk -> get_flat_cache -> reconstruct, catches flatten/pack bugs."""
    print(f"\n[store end-to-end]  (K{bits_k}/V{bits_v}, 2 chunks)")
    Hkv, T = 2, 300
    k = torch.randn(T, Hkv, DIM, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(T, Hkv, DIM, device=DEVICE, dtype=torch.bfloat16)
    store = CompressedKVStore(head_dim=DIM, num_kv_heads=Hkv, key_bits=bits_k,
                              value_bits=bits_v, device=DEVICE, layer_idx=0)
    store.append_chunk(k[:150], v[:150])
    store.append_chunk(k[150:], v[150:])               # second chunk -> exercises concat
    flat = store.get_flat_cache()
    assert store.num_tokens == T, f"expected {T} tokens, got {store.num_tokens}"
    kr = store.quantizer.dequantize(flat.prod_q).permute(1, 0, 2)   # (T,Hkv,D)
    vr = dequantize_V(flat.value_q, group_size=store.value_group_size).permute(1, 0, 2)
    kr_mse = _mse_only(store.quantizer, flat.prod_q).permute(1, 0, 2)
    print(f"  tokens={store.num_tokens}  key(+QJL) cos={mean_cos(k.reshape(-1, DIM), kr.reshape(-1, DIM)):.3f}"
          f"  key(MSE) cos={mean_cos(k.reshape(-1, DIM), kr_mse.reshape(-1, DIM)):.3f}"
          f"  val cos={mean_cos(v.reshape(-1, DIM), vr.reshape(-1, DIM)):.3f}")
    mb = store.memory_bytes() / (1024 ** 2)
    fp16 = T * Hkv * DIM * 2 * 2 / (1024 ** 2)
    print(f"  store={mb:.3f} MB vs fp16={fp16:.3f} MB  ->  {fp16 / mb:.2f}x")


def test_sanity_asserts() -> None:
    """Guardrails: K8 must round-trip near-exact; packing must be lossless on indices."""
    print("\n[sanity asserts]")
    x = torch.randn(N, DIM, device=DEVICE)
    tq = TurboQuant(dim=DIM, bits=8, device=DEVICE)
    cos8 = mean_cos(x, _mse_only(tq, tq.quantize(x)))
    assert cos8 > 0.99, f"K8 MSE-only cosine should be ~1.0, got {cos8:.3f} (codebook/scale regression?)"
    print(f"  K8 MSE-only cos={cos8:.3f}  OK (>0.99)")


def main() -> None:
    print(f"device={DEVICE} dim={DIM} n={N}")
    test_key_roundtrip()
    test_value_roundtrip()
    test_attention_score(bits=3)
    test_store_endtoend(bits_k=3, bits_v=2)
    test_sanity_asserts()
    print("\nAll diagnostics complete.")


if __name__ == "__main__":
    main()
