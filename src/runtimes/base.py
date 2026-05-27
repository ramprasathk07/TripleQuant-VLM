# src/runtimes/base.py
"""
RuntimeBase – abstract contract that every runtime must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from torch import Tensor
from src.config.schemas import BenchmarkModelEntry

class RuntimeBase(ABC):
    """
    Abstract base class for all model runtimes.

    Every concrete runtime (HFRuntime, VLLMRuntime, …) must implement
    all methods below. The runner always talks to this interface only.
    """

    name: str  # set by subclass, e.g. "hf", "vllm"

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self, entry: BenchmarkModelEntry) -> None:
        """
        Load the model + tokenizer described by entry.
        Applies quantization, device mapping, dtype, etc.
        Must be called before any inference method.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """
        Explicitly destroy the model, free GPU memory.
        Must call destroy_model_parallel() (vLLM), then gc.collect()
        and torch.cuda.empty_cache().
        """
        ...

    # ── Text generation ──────────────────────────────────────────────────────

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Greedy (temperature=0) or sampled generation.

        Args:
            prompts:        List of input strings.
            max_new_tokens: Token budget per prompt.
            temperature:    0.0 → greedy; > 0 → sampling.

        Returns:
            List of generated strings, one per prompt (decoded, no prompt echo).
        """
        ...

    @abstractmethod
    def generate_with_logits(
        self,
        prompts: list[str],
        max_new_tokens: int,
    ) -> Tuple[list[str], list[Tensor]]:
        """
        Generation that also returns per-step logit tensors.
        HF-only — VLLMRuntime raises NotImplementedError.

        Returns:
            (generated_texts, logits_list)
            logits_list[i] has shape (generated_len_i, vocab_size).
        """
        ...

    @abstractmethod
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """
        Single forward pass; returns full logit tensor.
        Used for perplexity and KL-divergence computation.
        HF-only — VLLMRuntime raises NotImplementedError.

        Args:
            input_ids: LongTensor of shape (1, seq_len).

        Returns:
            FloatTensor of shape (1, seq_len, vocab_size).
        """
        ...

    # ── Vision-Language generation ────────────────────────────────────────────

    @abstractmethod
    def generate_vlm(
        self,
        image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """
        Multimodal generation for VLMs (image + text → text).

        Args:
            image:          PIL Image or file path.
            prompt:         Text instruction.
            max_new_tokens: Token budget.

        Returns:
            Generated string (decoded, no prompt echo).

        Raises:
            NotImplementedError: If the loaded model is text-only.
        """
        ...

    # ── Latency / throughput profiling ───────────────────────────────────────

    @abstractmethod
    def measure_ttft_tpot(self, prompt: str, n: int = 100) -> dict:
        """
        Measures Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT).

        Runs n warm-up + timed trials.

        Returns:
            {
                "ttft_ms_mean":  float,
                "ttft_ms_p50":   float,
                "ttft_ms_p95":   float,
                "tpot_ms_mean":  float,
                "tpot_ms_p50":   float,
                "tpot_ms_p95":   float,
                "n_trials":      int,
            }
        """
        ...

    @abstractmethod
    def measure_throughput(
        self,
        prompt: str,
        batch_sizes: list[int],
        output_len: int,
    ) -> list[dict]:
        """
        Measures tokens/sec at different batch sizes.

        Args:
            prompt:      Single prompt string replicated across the batch.
            batch_sizes: List of batch sizes to test, e.g. [1, 4, 8, 16].
            output_len:  Fixed number of tokens to generate per sample.

        Returns:
            List of dicts, one per batch size:
            [
                {
                    "batch_size":       int,
                    "tokens_per_sec":   float,
                    "latency_ms":       float,
                    "oom":              bool,   # True if batch caused OOM
                },
                ...
            ]
        """
        ...

    # ── Memory ───────────────────────────────────────────────────────────────

    @abstractmethod
    def peak_vram_mb(self) -> float:
        """
        Returns the peak VRAM usage in MB since the model was loaded.
        Uses torch.cuda.max_memory_allocated() or equivalent.
        """
        ...

    # ── Convenience (non-abstract) ────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={getattr(self, 'name', '?')})"