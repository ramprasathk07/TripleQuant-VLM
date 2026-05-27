# src/runtimes/hf_runtime.py
"""
HFRuntime – HuggingFace Transformers backend.

Supports:
  - Text-only CausalLM models
  - VLM models (LLaVA-style via AutoProcessor)
  - 4-bit / 8-bit BitsAndBytes quantization
  - Multi-GPU via device_map="auto"
  - forward_logits() and generate_with_logits() for PPL / KL eval
  - TTFT / TPOT latency profiling
  - Batch throughput sweeps
  - Explicit unload with full GPU memory release
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Optional heavy imports (fail loudly only when actually used) ──────────────
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        LlavaForConditionalGeneration,
        GenerationConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .base import RuntimeBase


# ════════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════════

def _require_transformers() -> None:
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "HFRuntime requires `transformers`. "
            "Install it with: pip install transformers"
        )


def _require_pil() -> None:
    if not PIL_AVAILABLE:
        raise ImportError(
            "PIL is required for VLM image handling. "
            "Install it with: pip install pillow"
        )


def _build_bnb_config(quantization: Optional[str]) -> Optional["BitsAndBytesConfig"]:
    """
    Builds a BitsAndBytesConfig from a quantization string.

    Supported values:
        None / "none" / ""  → no quantization
        "int8"              → load_in_8bit
        "int4" / "nf4"      → load_in_4bit with nf4 + double quant
    """
    if not quantization or quantization.lower() in ("none", ""):
        return None

    _require_transformers()
    q = quantization.lower()

    if q == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)

    if q in ("int4", "nf4"):
        return BitsAndBytesConfig(
            load_in_4bit                = True,
            bnb_4bit_quant_type         = "nf4",
            bnb_4bit_use_double_quant   = True,
            bnb_4bit_compute_dtype      = torch.bfloat16,
        )

    raise ValueError(
        f"Unknown quantization '{quantization}'. "
        "Choose from: None, 'int8', 'int4', 'nf4'."
    )


def _resolve_dtype(dtype_str: Optional[str]) -> torch.dtype:
    """
    Maps a dtype string to a torch.dtype.

    Supported:
        "float32" / "fp32"   → torch.float32
        "float16" / "fp16"   → torch.float16
        "bfloat16" / "bf16"  → torch.bfloat16
        None / "auto"        → torch.bfloat16  (sensible default)
    """
    _map = {
        "float32":  torch.float32,
        "fp32":     torch.float32,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
    }
    if dtype_str is None or dtype_str.lower() == "auto":
        return torch.bfloat16
    try:
        return _map[dtype_str.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. "
            f"Choose from: {list(_map.keys())} or 'auto'."
        )


def _sync_cuda() -> None:
    """Synchronizes CUDA if available (for accurate timing)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(sorted_data: list[float], p: float) -> float:
    """Returns the p-th percentile from a pre-sorted list."""
    idx = int(len(sorted_data) * p)
    idx = max(0, min(idx, len(sorted_data) - 1))
    return sorted_data[idx]


def _timing_stats(times_ms: list[float]) -> dict:
    """Returns mean / p50 / p95 / p99 / min / max for a list of timings."""
    s = sorted(times_ms)
    return {
        "mean": statistics.mean(s),
        "p50":  _percentile(s, 0.50),
        "p95":  _percentile(s, 0.95),
        "p99":  _percentile(s, 0.99),
        "min":  s[0],
        "max":  s[-1],
    }


# ════════════════════════════════════════════════════════════════════════════════
# HFRuntime
# ════════════════════════════════════════════════════════════════════════════════

class HFRuntime(RuntimeBase):
    """
    HuggingFace Transformers runtime.

    Usage:
        runtime = HFRuntime()
        runtime.load(entry)          # loads model + tokenizer
        outputs = runtime.generate(["Hello!"], max_new_tokens=64)
        runtime.unload()             # frees all GPU memory
    """

    name = "hf"

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        _require_transformers()

        self.model:      Optional["AutoModelForCausalLM"] = None
        self.tokenizer:  Optional["AutoTokenizer"]        = None
        self.processor:  Optional["AutoProcessor"]        = None

        self._is_vlm:    bool          = False
        self._model_id:  str           = ""
        self._device:    torch.device  = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ════════════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ════════════════════════════════════════════════════════════════════════════

    def load(self, entry: "BenchmarkModelEntry") -> None:
        """
        Loads model + tokenizer / processor from a BenchmarkModelEntry.

        Reads from entry:
            entry.path                str   – HF hub id or local dir
            entry.model_type          str   – 'vlm' => load as VLM
            entry.hf_quantization     Optional[str] – None / 'int8' / 'int4' / 'nf4'
            entry.dtype               str   – 'auto' / 'bfloat16' / 'float16' / ...
            entry.device_map          str   – 'auto' / 'cuda:0' / ...
            entry.trust_remote_code   bool

        Note:
            Pre-quantized (compressed-tensors / ModelOpt) checkpoints load
            directly via from_pretrained — leave hf_quantization=None for those.
            hf_quantization is only for on-the-fly BitsAndBytes quantization.

        Raises:
            ImportError:  if transformers is not installed.
            RuntimeError: if model is already loaded.
        """
        if self.model is not None:
            raise RuntimeError(
                "A model is already loaded. Call unload() before loading a new one."
            )

        model_id       = entry.path
        self._model_id = model_id

        quant_cfg    = _build_bnb_config(getattr(entry, "hf_quantization", None))
        dtype        = _resolve_dtype(getattr(entry, "dtype", None))
        device_map   = getattr(entry, "device_map",        "auto")
        trust_rc     = getattr(entry, "trust_remote_code", False)
        self._is_vlm = entry.is_vlm

        # Reset VRAM peak counter before loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info(
            f"[HFRuntime] Loading '{model_id}' | "
            f"vlm={self._is_vlm} | quant={getattr(entry, 'hf_quantization', None)} | "
            f"dtype={dtype} | device_map={device_map}"
        )

        if self._is_vlm:
            self._load_vlm(model_id, quant_cfg, dtype, device_map, trust_rc)
        else:
            self._load_causal_lm(model_id, quant_cfg, dtype, device_map, trust_rc)

        logger.info(f"[HFRuntime] '{model_id}' ready. "
                    f"Peak VRAM so far: {self.peak_vram_mb():.1f} MB")

    def _load_causal_lm(
        self,
        model_id:   str,
        quant_cfg,
        dtype:      torch.dtype,
        device_map: str,
        trust_rc:   bool,
    ) -> None:
        """Loads a standard CausalLM model and its tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast          = True,
            trust_remote_code = trust_rc,
        )
        # Ensure pad token exists (required for batched generation)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Decoder-only models require LEFT padding for correct batched generation;
        # right padding corrupts outputs and breaks prompt-length slicing.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config = quant_cfg,
            torch_dtype         = dtype,
            device_map          = device_map,
            trust_remote_code   = trust_rc,
        )
        self.model.eval()

    def _load_vlm(
        self,
        model_id:   str,
        quant_cfg,
        dtype:      torch.dtype,
        device_map: str,
        trust_rc:   bool,
    ) -> None:
        """
        Loads a LLaVA-style VLM via AutoProcessor.
        Falls back to AutoModelForCausalLM if LlavaForConditionalGeneration
        is unavailable for the requested model.
        """
        _require_pil()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code = trust_rc,
        )
        self.tokenizer = self.processor.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # Try LLaVA first, fall back to generic CausalLM
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config = quant_cfg,
                torch_dtype         = dtype,
                device_map          = device_map,
                trust_remote_code   = trust_rc,
            )
        except (ValueError, OSError):
            logger.warning(
                f"[HFRuntime] LlavaForConditionalGeneration failed for '{model_id}'. "
                "Falling back to AutoModelForCausalLM."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config = quant_cfg,
                torch_dtype         = dtype,
                device_map          = device_map,
                trust_remote_code   = trust_rc,
            )

        self.model.eval()

    def unload(self) -> None:
        """
        Destroys the model and tokenizer, frees all GPU memory.
        Safe to call multiple times.
        """
        logger.info(f"[HFRuntime] Unloading '{self._model_id}' …")

        for attr in ("model", "tokenizer", "processor"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)

        self._model_id = ""
        self._is_vlm   = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("[HFRuntime] Unload complete.")

    # ════════════════════════════════════════════════════════════════════════════
    # Text generation
    # ════════════════════════════════════════════════════════════════════════════

    def generate(
        self,
        prompts:        list[str],
        max_new_tokens: int,
        temperature:    float = 0.0,
    ) -> list[str]:
        """
        Batched text generation (greedy or sampled).

        Args:
            prompts:        Input strings (batch).
            max_new_tokens: Token budget per prompt.
            temperature:    0.0 → greedy; > 0.0 → multinomial sampling.

        Returns:
            List of decoded strings (prompt not included).
        """
        self._check_loaded()

        inputs = self.tokenizer(
            prompts,
            return_tensors  = "pt",
            padding         = True,
            truncation      = True,
        ).to(self._device)

        do_sample  = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens  = max_new_tokens,
            do_sample       = do_sample,
            pad_token_id    = self.tokenizer.pad_token_id,
            eos_token_id    = self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            for ids in output_ids
        ]

    def generate_with_logits(
        self,
        prompts:        list[str],
        max_new_tokens: int,
    ) -> Tuple[list[str], list[Tensor]]:
        """
        Greedy generation that also returns per-step logit tensors.

        Returns:
            texts:       List of decoded output strings (batch).
            logits_list: List of tensors, one per batch item.
                         Each tensor has shape (generated_len, vocab_size).

        Note:
            HF-only — VLLMRuntime raises NotImplementedError for this method.
        """
        self._check_loaded()

        inputs = self.tokenizer(
            prompts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
        ).to(self._device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens          = max_new_tokens,
                do_sample               = False,
                return_dict_in_generate = True,
                output_scores           = True,
                pad_token_id            = self.tokenizer.pad_token_id,
                eos_token_id            = self.tokenizer.eos_token_id,
            )

        # outputs.scores: tuple of (batch_size, vocab_size), one per step
        texts = [
            self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in outputs.sequences
        ]

        # Stack → (steps, batch, vocab) → split per item → (steps, vocab)
        if outputs.scores:
            stacked      = torch.stack(outputs.scores, dim=0)  # (T, B, V)
            logits_list  = [stacked[:, i, :] for i in range(stacked.size(1))]
        else:
            logits_list  = [torch.empty(0) for _ in prompts]

        return texts, logits_list

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """
        Single forward pass without generation.
        Returns the full logit tensor over the entire input sequence.

        Args:
            input_ids: LongTensor of shape (1, seq_len).

        Returns:
            FloatTensor of shape (1, seq_len, vocab_size).

        Used by:
            compute_ppl()     in eval_llm.py
            eval_logit_kl()   in eval_llm.py
        """
        self._check_loaded()
        input_ids = input_ids.to(self._device)

        with torch.no_grad():
            out = self.model(input_ids=input_ids)

        return out.logits   # (1, seq_len, vocab_size)

    # ════════════════════════════════════════════════════════════════════════════
    # VLM generation
    # ════════════════════════════════════════════════════════════════════════════

    def generate_vlm(
        self,
        image,
        prompt:         str,
        max_new_tokens: int,
    ) -> str:
        """
        Multimodal generation for LLaVA-style VLMs.

        Args:
            image:          PIL.Image.Image or path string.
            prompt:         Text instruction / question.
            max_new_tokens: Token budget.

        Returns:
            Decoded generated string (prompt not included).

        Raises:
            NotImplementedError: If model was not loaded as a VLM.
            ImportError:         If Pillow is not installed.
        """
        self._check_loaded()

        if not self._is_vlm:
            raise NotImplementedError(
                "generate_vlm() requires a VLM. "
                "Set entry.is_vlm=True when calling load()."
            )

        _require_pil()

        # Ensure we have a PIL image
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        elif not isinstance(image, PILImage.Image):
            raise TypeError(
                f"image must be a PIL.Image or a file path string, "
                f"got {type(image).__name__}."
            )

        inputs = self.processor(
            text          = prompt,
            images        = image,
            return_tensors = "pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                pad_token_id   = self.tokenizer.pad_token_id,
                eos_token_id   = self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        )

    # ════════════════════════════════════════════════════════════════════════════
    # Latency profiling
    # ════════════════════════════════════════════════════════════════════════════

    def measure_ttft_tpot(self, prompt: str, n: int = 100) -> dict:
        """
        Measures Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT)
        over n timed trials.

        Strategy:
            - TTFT  : time for a single forward pass (prompt → 1 new token).
            - TPOT  : (total_time - ttft) / (output_tokens - 1) averaged
                      over a fixed 64-token generation.

        Args:
            prompt: Input string used for all trials.
            n:      Number of timed repetitions (warm-up: first 5 discarded).

        Returns:
            {
                "ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
                "ttft_ms_min",  "ttft_ms_max",
                "tpot_ms_mean", "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
                "tpot_ms_min",  "tpot_ms_max",
                "n_trials": int,
            }
        """
        self._check_loaded()

        OUTPUT_TOKENS = 64
        WARMUP        = min(5, n)

        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self._device)

        ttft_times: list[float] = []
        tpot_times: list[float] = []

        for trial in range(n + WARMUP):
            # ── TTFT : single greedy step ─────────────────────────────────────
            _sync_cuda()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens = 1,
                    do_sample      = False,
                    pad_token_id   = self.tokenizer.pad_token_id,
                )
            _sync_cuda()
            ttft_ms = (time.perf_counter() - t0) * 1_000

            # ── TPOT : full generation, then back-calculate per-token time ────
            _sync_cuda()
            t1 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens          = OUTPUT_TOKENS,
                    do_sample               = False,
                    return_dict_in_generate = True,
                    output_scores           = True,
                    pad_token_id            = self.tokenizer.pad_token_id,
                )
            _sync_cuda()
            total_ms    = (time.perf_counter() - t1) * 1_000
            n_generated = len(out.scores)   # actual tokens produced

            if n_generated > 1:
                tpot_ms = (total_ms - ttft_ms) / (n_generated - 1)
            else:
                tpot_ms = total_ms   # edge case: only 1 token generated

            # Discard warm-up trials
            if trial >= WARMUP:
                ttft_times.append(ttft_ms)
                tpot_times.append(max(tpot_ms, 0.0))

        ttft_stats = _timing_stats(ttft_times)
        tpot_stats = _timing_stats(tpot_times)

        return {
            "ttft_ms_mean": ttft_stats["mean"],
            "ttft_ms_p50":  ttft_stats["p50"],
            "ttft_ms_p95":  ttft_stats["p95"],
            "ttft_ms_p99":  ttft_stats["p99"],
            "ttft_ms_min":  ttft_stats["min"],
            "ttft_ms_max":  ttft_stats["max"],
            "tpot_ms_mean": tpot_stats["mean"],
            "tpot_ms_p50":  tpot_stats["p50"],
            "tpot_ms_p95":  tpot_stats["p95"],
            "tpot_ms_p99":  tpot_stats["p99"],
            "tpot_ms_min":  tpot_stats["min"],
            "tpot_ms_max":  tpot_stats["max"],
            "n_trials":     n,
        }

    # ════════════════════════════════════════════════════════════════════════════
    # Throughput profiling
    # ════════════════════════════════════════════════════════════════════════════

    def measure_throughput(
        self,
        prompt:      str,
        batch_sizes: list[int],
        output_len:  int,
    ) -> list[dict]:
        """
        Sweeps batch sizes and records tokens/sec, latency, and OOM status.

        Each batch_size is tested with 3 timed runs; the median latency is
        reported to reduce noise.

        Args:
            prompt:      Single prompt string replicated across the batch.
            batch_sizes: List of batch sizes to test, e.g. [1, 4, 8, 16].
            output_len:  Fixed number of tokens to generate per sample.

        Returns:
            List of dicts ordered by batch_size:
            [
                {
                    "batch_size":     int,
                    "tokens_per_sec": float,
                    "latency_ms":     float,   # median across runs
                    "total_tokens":   int,
                    "oom":            bool,
                },
                ...
            ]
        """
        self._check_loaded()

        RUNS_PER_BATCH = 3
        results        = []

        for bs in batch_sizes:
            prompts = [prompt] * bs
            inputs  = self.tokenizer(
                prompts,
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
            ).to(self._device)

            oom          = False
            latencies_ms = []

            for run in range(RUNS_PER_BATCH):
                try:
                    _sync_cuda()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        self.model.generate(
                            **inputs,
                            max_new_tokens = output_len,
                            do_sample      = False,
                            pad_token_id   = self.tokenizer.pad_token_id,
                        )
                    _sync_cuda()
                    latencies_ms.append((time.perf_counter() - t0) * 1_000)

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"[HFRuntime] OOM at batch_size={bs}. "
                        "Skipping remaining runs for this batch size."
                    )
                    oom = True
                    torch.cuda.empty_cache()
                    break

            if oom or not latencies_ms:
                results.append({
                    "batch_size":     bs,
                    "tokens_per_sec": 0.0,
                    "latency_ms":     float("inf"),
                    "total_tokens":   0,
                    "oom":            True,
                })
                continue

            # Use median latency across runs for stability
            median_latency_ms = sorted(latencies_ms)[RUNS_PER_BATCH // 2]
            total_tokens      = bs * output_len
            tokens_per_sec    = total_tokens / (median_latency_ms / 1_000)

            results.append({
                "batch_size":     bs,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "latency_ms":     round(median_latency_ms, 2),
                "total_tokens":   total_tokens,
                "oom":            False,
            })

            logger.info(
                f"[HFRuntime] bs={bs:>3} | "
                f"tps={tokens_per_sec:>8.1f} | "
                f"latency={median_latency_ms:>7.1f} ms"
            )

        return results

    # ════════════════════════════════════════════════════════════════════════════
    # Memory
    # ════════════════════════════════════════════════════════════════════════════

    def peak_vram_mb(self) -> float:
        """
        Returns peak VRAM allocated (in MB) since last reset.
        Returns 0.0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def current_vram_mb(self) -> float:
        """
        Returns currently allocated VRAM in MB.
        Useful for monitoring between eval steps.
        """
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 2)

    # ════════════════════════════════════════════════════════════════════════════
    # Extras: score_choices (used by eval_mmlu_tiny)
    # ════════════════════════════════════════════════════════════════════════════

    def score_choices(
        self,
        prompt:  str,
        choices: list[str],
    ) -> dict[str, float]:
        """
        Returns a dict mapping each choice label to its log-probability
        as the next token after the prompt.

        Args:
            prompt:  Full prompt string ending with "Answer:".
            choices: List of single-token labels, e.g. ["A", "B", "C", "D"].

        Returns:
            {"A": -0.32, "B": -1.45, "C": -2.10, "D": -3.88}

        Used by:
            _score_mmlu_choices() in eval_llm.py
        """
        self._check_loaded()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"]).logits  # (1, T, V)

        last_log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # (V,)

        scores = {}
        for label in choices:
            token_ids = self.tokenizer.encode(
                f" {label}", add_special_tokens=False
            )
            if not token_ids:
                scores[label] = float("-inf")
                continue
            scores[label] = last_log_probs[token_ids[0]].item()

        return scores

    # ════════════════════════════════════════════════════════════════════════════
    # Internal guards
    # ════════════════════════════════════════════════════════════════════════════

    def _check_loaded(self) -> None:
        """Raises RuntimeError if load() has not been called yet."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "HFRuntime: model is not loaded. Call load(entry) first."
            )

    # ════════════════════════════════════════════════════════════════════════════
    # Dunder helpers
    # ════════════════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        status = f"loaded='{self._model_id}'" if self.model else "unloaded"
        return f"HFRuntime({status}, device={self._device}, vlm={self._is_vlm})"

    def __enter__(self) -> "HFRuntime":
        """Supports usage as a context manager."""
        return self

    def __exit__(self, *_) -> None:
        """Automatically unloads on context manager exit."""
        self.unload()