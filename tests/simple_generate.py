"""
tests/simple_generate.py — test / interactive runner for quantized causal LMs.

Usage examples:
    # Run built-in test prompts on a compressed model
    python tests/simple_generate.py --model ./output/TinyLlama-W4A16

    # Compare quantized vs FP16 baseline
    python tests/simple_generate.py --model ./output/TinyLlama-W4A16 \
                                    --baseline TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Interactive chat
    python tests/simple_generate.py --model ./output/TinyLlama-W4A16 --interactive

    # Plain uncompressed model
    python tests/simple_generate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                                    --no-compressed
"""

from __future__ import annotations

# compressed_tensors MUST be imported before transformers loads any model.
# It patches AutoConfig and modeling_utils so pack-quantized checkpoints work.
try:
    import compressed_tensors  # noqa: F401
except ImportError:
    pass  # clean error raised later if --compressed is used

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_compressed(model_path: str) -> bool:
    """Auto-detect if checkpoint was saved with save_compressed=True."""
    cfg = Path(model_path) / "config.json"
    if not cfg.exists():
        return False
    try:
        data = json.loads(cfg.read_text())
        fmt = data.get("quantization_config", {}).get("format", "")
        return fmt == "pack-quantized"
    except Exception:
        return False


def disk_size(model_path: str) -> str:
    path = Path(model_path)
    total = sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.suffix in {".safetensors", ".bin", ".pt"}
    )
    mb = total / (1024 ** 2)
    return f"{mb:.1f} MB" if mb < 1024 else f"{mb / 1024:.2f} GB"


def vram_usage() -> str:
    if not torch.cuda.is_available():
        return "N/A (no CUDA)"
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    res   = torch.cuda.memory_reserved()  / (1024 ** 2)
    return f"alloc={alloc:.0f} MB  reserved={res:.0f} MB"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    compressed: bool | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal LM and its tokenizer.

    compressed=None  → auto-detect from config.json
    compressed=True  → require compressed_tensors hooks (pack-quantized)
    compressed=False → plain from_pretrained
    """
    if compressed is None:
        compressed = _detect_compressed(model_path)

    print(f"\n Loading  : {model_path}")
    print(f"   Format   : {'compressed' if compressed else 'plain FP'}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if compressed:
        try:
            import compressed_tensors  # noqa: F401 — activates hooks
        except ImportError:
            raise ImportError(
                "compressed-tensors is required for pack-quantized models.\n"
                "Install: pip install llmcompressor"
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )

    model.eval()
    elapsed = time.perf_counter() - t0

    print(f"   Load time: {elapsed:.1f}s")
    print(f"   Disk size: {disk_size(model_path)}")
    print(f"   VRAM     : {vram_usage()}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_ids  = output_ids[0][input_len:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    n_tok    = len(new_ids)
    print(f"   [Perf] {n_tok} tokens / {elapsed:.2f}s  ({n_tok / elapsed:.1f} tok/s)  VRAM: {vram_usage()}")
    return response


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

_TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is in simple terms.",
    "Write a short Python function to reverse a string.",
    "What are the pros and cons of model quantization?",
]


def run_test_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 200,
) -> None:
    sep = "-" * 70
    print(f"\n{sep}\n  TEST PROMPTS\n{sep}")
    for i, prompt in enumerate(_TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(_TEST_PROMPTS)}] {prompt}")
        response = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        print(f"    {response}")
    print(f"\n{sep}")


def run_comparison(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    max_new_tokens: int = 200,
) -> None:
    prompts = [
        "Explain quantization in one paragraph.",
        "Write a haiku about artificial intelligence.",
    ]
    sep = "-" * 70
    print(f"\n{sep}\n  BASELINE COMPARISON\n{sep}")
    for prompt in prompts:
        print(f"\n> {prompt}")
        print("  [Quantized] ", end="", flush=True)
        r_q = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        print(f"  {r_q}")
        print("  [Baseline]  ", end="", flush=True)
        r_b = generate(base_model, base_tokenizer, prompt, max_new_tokens=max_new_tokens)
        print(f"  {r_b}")
    print(f"\n{sep}")


def run_interactive(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 200,
) -> None:
    sep = "-" * 70
    print(f"\n{sep}\n  INTERACTIVE  (type 'exit' to quit)\n{sep}")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        response = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        print(f"Model: {response}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test a quantized (or plain) causal LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Path to model dir or HF model ID.")
    p.add_argument("--baseline", default=None,
                   help="Optional FP16 baseline model path/ID for comparison.")
    p.add_argument("--compressed", dest="compressed", action="store_true", default=None,
                   help="Force compressed-tensors loading mode.")
    p.add_argument("--no-compressed", dest="compressed", action="store_false",
                   help="Force plain from_pretrained (skip compressed_tensors hooks).")
    p.add_argument("--baseline-compressed", action="store_true", default=False,
                   help="Baseline is also a compressed checkpoint.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--skip-tests", action="store_true",
                   help="Skip the built-in test-prompt suite.")
    p.add_argument("--interactive", action="store_true",
                   help="Launch interactive chat after tests.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    model, tokenizer = load_model(args.model, compressed=args.compressed)

    if not args.skip_tests:
        run_test_prompts(model, tokenizer, max_new_tokens=args.max_new_tokens)

    if args.baseline:
        base_model, base_tokenizer = load_model(
            args.baseline, compressed=args.baseline_compressed
        )
        run_comparison(
            model, tokenizer,
            base_model, base_tokenizer,
            max_new_tokens=args.max_new_tokens,
        )

    if args.interactive:
        run_interactive(model, tokenizer, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
