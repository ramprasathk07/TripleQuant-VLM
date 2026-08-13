# Your quantized model isn't slow. Your runtime is.

I quantized Qwen3-1.7B to 4-bit and it got **five times slower**. Then I ran the exact
same file on a different runtime and it got faster than the original.

Same weights. Same GPU. Same prompt. 13x apart.

![Same checkpoint, two runtimes](plots/runtime_gap.png)

If you take one thing from this: **a quantized checkpoint has no inherent speed.** It has
a speed *on a runtime*. Benchmark it on the wrong one and you'll conclude the format is
garbage, when what you measured was your inference loop.

---

## The setup

RTX 3060 (12 GB, Ampere), Qwen3-1.7B, batch size 1, greedy decode. One checkpoint —
AWQ-W4A16 produced by `llm-compressor` — loaded two ways:

- **HuggingFace `generate()`**, eager mode, the default thing everyone reaches for
- **vLLM 0.11.2**, which ships fused Marlin kernels for `compressed-tensors` weights

| Checkpoint | HF eager | vLLM | Ratio |
|---|---|---|---|
| FP16 baseline | 21.5 tok/s | 56.1 tok/s | 2.6x |
| **AWQ-W4A16** | **4.3 tok/s** | **57.9 tok/s** | **13.6x** |
| GPTQ-W8A8 | 2.6 tok/s | 49.3 tok/s | 18.8x |

Look at the AWQ row twice. Under HF eager it's **5x slower than the FP16 model it was
supposed to accelerate**. Under vLLM it's *faster than FP16* — 57.9 vs 56.1.

---

## Why 4-bit weights make eager mode slower

The intuition "fewer bits = less memory traffic = faster" is correct about the *weights*
and wrong about *what actually executes*.

Decode at batch 1 is memory-bound. You're streaming an entire weight matrix through the
GPU to produce one token — almost no arithmetic per byte moved. So bytes moved sets your
speed.

A W4A16 checkpoint stores `qweight` (two 4-bit values per byte) plus a `scale` per group
of 128. Your GPU cannot multiply by a 4-bit integer. Somewhere, those weights become
bf16. The question is only *where*.

**HF eager path, per linear layer:**

```
read packed int4 from HBM      (small — this is the win you wanted)
unpack + scale → bf16
write bf16 tensor back to HBM  ← the entire saving, spent
matmul reads that bf16 tensor back from HBM
```

You pay a full round-trip through HBM to materialize a dequantized tensor that fp16 never
had to write, because fp16 was already in the right format. You bought a smaller file and
paid for it in bandwidth — the one resource that was your bottleneck.

**Marlin's path:** the weights stay packed all the way into the matmul's inner loop.
Dequantization happens in registers, per tile, fused into the multiply. The bf16 tensor is
never written anywhere. Now the 4x-fewer-bytes actually reaches the metric you care about.

That's the whole 13x. Not a different algorithm — the same arithmetic, with one HBM
round-trip removed.

---

## The memory saving was real the whole time

Worth being precise, because "4-bit was slower" is not the same as "4-bit did nothing":

| | Resident VRAM |
|---|---|
| FP16 | 3.28 GB |
| AWQ-W4A16 | **1.30 GB** (−60%) |

The compression is genuine and shows up on *both* runtimes. Only the speed depended on
kernels. If you're quantizing to fit a model on a smaller card, eager mode serves you
fine. If you're quantizing for throughput, the runtime is the entire game.

---

## What this means for your benchmarks

1. **Never report "format X is slow" from an eager-mode measurement.** You measured your
   runtime's kernel coverage.
2. **Check whether your runtime has kernels for your format** before choosing the format.
   `compressed-tensors` → vLLM has Marlin. torchao int4 → check which packing formats have
   kernels on *your* GPU (I hit one that required an uninstallable package).
3. **Separate the two claims.** "Uses less memory" and "runs faster" are different
   properties with different evidence. Quantization always delivers the first. The second
   is conditional.

The most expensive version of this mistake is picking your quantization format based on a
speed benchmark you ran on the wrong runtime — and then shipping the loser.

---

*Numbers from [TripleQuant-VLM](https://github.com/ramprasathk07/TripleQuant-VLM), RTX 3060 12 GB, driver
610.62 / CUDA 12.8 / torch 2.8.0, vLLM 0.11.2 in WSL2. Raw JSON and the harness that
produced them are in the repo (`docs/qwen3_1_7b_leaderboard.md`). vLLM was run in eager
mode (CUDA-graph capture OOMed the memory-capped VM), which if anything *understates* the
gap.*
