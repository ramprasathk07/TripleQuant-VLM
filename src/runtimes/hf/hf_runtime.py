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
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
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

from ..base import RuntimeBase

# On Windows/WDDM, CUDA allocations can silently spill into shared system RAM
# instead of raising OutOfMemoryError — torch.cuda.max_memory_allocated() then
# reports a peak *larger than the card's physical VRAM*, and a context-length
# probe reads as "fits" when it actually paged at PCIe speed. 0.95 of the
# hardware total (a static, driver-independent fact) is treated as the real
# capacity ceiling; anything above it is flagged rather than trusted.
_VRAM_OVERSUBSCRIBE_SAFETY = 0.95



# Internal helpers
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
    """Build a BitsAndBytesConfig for on-the-fly quantization.

    Args:
        quantization: Quantization method (None/'none'/'', 'int8', 'int4', 'nf4').

    Returns:
        BitsAndBytesConfig for load_in_8bit or load_in_4bit, or None for no quantization.

    Raises:
        ValueError: If quantization string is not recognized.
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
    """Map dtype string to torch.dtype constant.

    Args:
        dtype_str: Dtype name or None/'auto' for default (bfloat16).
                  Supported: 'float32'/'fp32', 'float16'/'fp16', 'bfloat16'/'bf16'.

    Returns:
        torch.dtype constant.

    Raises:
        ValueError: If dtype_str is not recognized.
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
    """Synchronize CUDA device for accurate timing measurements."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile from a pre-sorted list.

    Args:
        sorted_data: List sorted in ascending order.
        p: Percentile rank (0.0-1.0, e.g., 0.95 for 95th percentile).

    Returns:
        Value at or near the requested percentile.
    """
    idx = int(len(sorted_data) * p)
    idx = max(0, min(idx, len(sorted_data) - 1))
    return sorted_data[idx]


def _timing_stats(times_ms: list[float]) -> dict:
    """Compute timing statistics (mean, percentiles, min/max).

    Args:
        times_ms: List of timing measurements in milliseconds.

    Returns:
        Dict with keys: mean, p50, p95, p99, min, max.
    """
    s = sorted(times_ms)
    return {
        "mean": statistics.mean(s),
        "p50":  _percentile(s, 0.50),
        "p95":  _percentile(s, 0.95),
        "p99":  _percentile(s, 0.99),
        "min":  s[0],
        "max":  s[-1],
    }
import json
from pathlib import Path

def _detect_compressed(model_path: str) -> bool:
    """Detect if a checkpoint was saved with save_compressed=True.

    Checks the quantization_config.format field in config.json for 'pack-quantized'.

    Args:
        model_path: Path to model checkpoint directory.

    Returns:
        True if compressed format detected; False otherwise.
    """
    cfg = Path(model_path) / "config.json"
    if not cfg.exists():
        return False
    try:
        data = json.loads(cfg.read_text())
        fmt = data.get("quantization_config", {}).get("format", "")
        return fmt == "pack-quantized"
    except Exception:
        return False


def _model_load_kwargs(quant_cfg, dtype, device_map, trust_rc: bool) -> dict:
    """Build from_pretrained kwargs, omitting quantization_config when None.

    Passing ``quantization_config=None`` explicitly OVERRIDES a pre-quantized
    checkpoint's own config with None, which then crashes transformers'
    quantizer detection (`supports_quant_method(None)` -> AttributeError). Only
    include the key for genuine on-the-fly BitsAndBytes quantization.
    """
    kw = dict(torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_rc)
    if quant_cfg is not None:
        kw["quantization_config"] = quant_cfg
    return kw

_QC_REPR_PATCHED = False

def _patch_quantization_config_repr() -> None:
    """Work around a transformers bug that crashes loading compressed-tensors VLMs.

    When `from_pretrained` loads a quantized composite VLM (e.g. Qwen2.5-VL), it
    sets `quantization_config = None` on the *sub*-config (text_config) and then
    logs `logger.info(f"Model config {config}")`. The repr calls both
    `to_dict()` and `to_diff_dict()`, each doing `self.quantization_config.to_dict()`
    without a None-guard -> `AttributeError: 'NoneType' object has no attribute
    'to_dict'`. Only triggers when transformers logging is at INFO (the benchmark
    sets root logging to INFO).

    Wrap both methods to temporarily drop a None `quantization_config` so the buggy
    branch is skipped, then restore it. Idempotent; no-op if transformers missing.
    """
    global _QC_REPR_PATCHED
    if _QC_REPR_PATCHED or not TRANSFORMERS_AVAILABLE:
        return
    import transformers.configuration_utils as _cu

    def _guard(orig):
        def _wrapped(self):
            if getattr(self, "quantization_config", "x") is None and \
                    "quantization_config" in self.__dict__:
                saved = self.__dict__.pop("quantization_config")
                try:
                    return orig(self)
                finally:
                    self.__dict__["quantization_config"] = saved
            return orig(self)
        return _wrapped

    # Both repr paths (to_dict / to_diff_dict) call quantization_config.to_dict().
    for _name in ("to_dict", "to_diff_dict"):
        if hasattr(_cu.PretrainedConfig, _name):
            setattr(_cu.PretrainedConfig, _name, _guard(getattr(_cu.PretrainedConfig, _name)))
    _QC_REPR_PATCHED = True



# HFRuntime
class HFRuntime(RuntimeBase):
    """HuggingFace Transformers backend for LLM and VLM inference.

    Supports text-only LLMs and vision-language models (VLMs) with optional
    on-the-fly BitsAndBytes quantization (int8, int4). Provides logit access
    for perplexity/KL-divergence evaluation, latency profiling (TTFT/TPOT),
    and throughput measurements. Explicit unload() frees all GPU memory.

    Attributes:
        name: Identifier 'hf' for runtime registry.
        model: Loaded model (AutoModelForCausalLM or VLM equivalent).
        tokenizer: Loaded tokenizer (AutoTokenizer).
        processor: Loaded processor (AutoProcessor for VLMs).

    Usage:
        runtime = HFRuntime()
        runtime.load(entry)
        outputs = runtime.generate(["Hello!"], max_new_tokens=64)
        runtime.unload()
    """

    name = "hf"

    def __init__(self) -> None:
        """Initialize HFRuntime, checking transformers availability."""
        _require_transformers()

        self.model:      Optional["AutoModelForCausalLM"] = None
        self.tokenizer:  Optional["AutoTokenizer"]        = None
        self.processor:  Optional["AutoProcessor"]        = None

        self._is_vlm:    bool          = False
        self._model_id:  str           = ""
        self._model_vram_mb: float     = 0.0    # resident weights memory after load
        self._tq_cfg                   = None   # CacheConfig when TurboQuant enabled
        self._device:    torch.device  = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _maybe_tq_cache(self, batch_size: int):
        """Build a fresh TurboQuant cache for a single-sequence generate, or None.

        TurboQuant's cache assumes batch size 1 (it indexes ``key_states[0]``), so
        batched calls fall back to the model's default cache. A new cache is built
        per call because it accumulates state across the generation.
        """
        if self._tq_cfg is None or batch_size != 1:
            return None
        from .cache import TurboQuantCache
        return TurboQuantCache(self.model.config, self._tq_cfg)

    def _tq_gen_kwargs(self, batch_size: int) -> dict:
        """generate() kwargs that route through a fresh TurboQuant cache, or {}."""
        cache = self._maybe_tq_cache(batch_size)
        return {"past_key_values": cache, "use_cache": True} if cache is not None else {}

    def load(self, entry: "BenchmarkModelEntry") -> None:
        """Load model and tokenizer/processor from Hugging Face.

        Routes to VLM (AutoProcessor + model) or LLM (AutoTokenizer + model)
        loading based on entry.model_type. Applies BitsAndBytes quantization
        if entry.hf_quantization is set; otherwise loads pre-quantized checkpoints
        (compressed-tensors, ModelOpt) directly.

        Args:
            entry: BenchmarkModelEntry specifying model location, dtype, device_map,
                   quantization, and model type.

        Raises:
            ImportError: If transformers is not installed.
            RuntimeError: If model is already loaded or loading fails.
        """
        if self.model is not None:
            raise RuntimeError(
                "A model is already loaded. Call unload() before loading a new one."
            )

        _patch_quantization_config_repr()  # tolerate compressed-tensors VLM config repr

        model_id       = entry.path
        self._model_id = model_id

        quant_cfg    = _build_bnb_config(getattr(entry, "hf_quantization", None))
        dtype        = _resolve_dtype(getattr(entry, "dtype", None))
        device_map   = getattr(entry, "device_map",        "auto")
        trust_rc     = getattr(entry, "trust_remote_code", False)

        # Resolve which loader to use. Explicit `model_class` wins; "auto" falls back
        # to the model_type heuristic (vlm -> VLM loader, else causal-LM).
        model_class  = getattr(entry, "model_class", "auto")
        route        = self._resolve_load_route(model_class, entry.is_vlm)
        self._is_vlm = route == "vlm"

        # Reset VRAM peak counter before loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info(
            f"[HFRuntime] Loading '{model_id}' | route={route} | model_class={model_class} | "
            f"quant={getattr(entry, 'hf_quantization', None)} | "
            f"dtype={dtype} | device_map={device_map}"
        )

        if route == "vlm":
            self._load_vlm(model_id, quant_cfg, dtype, device_map, trust_rc)
        else:
            self._load_causal_lm(model_id, quant_cfg, dtype, device_map, trust_rc,
                                 seq2seq=(route == "seq2seq"))

        if getattr(entry, "torchao", None):
            self._apply_torchao(entry.torchao)
        self._sanitize_generation_config()
        self._maybe_enable_turboquant(entry)

        # Record resident weights memory, then reset the peak counter so the perf
        # metrics measure INFERENCE memory rather than the transient bf16 load peak —
        # this is what reveals weight-quantization savings (torchao int8/int4).
        if torch.cuda.is_available():
            self._model_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        logger.info(f"[HFRuntime] '{model_id}' ready. "
                    f"Resident weights VRAM: {self._model_vram_mb:.1f} MB")

    def model_vram_mb(self) -> float:
        """Resident GPU memory (MB) held by the loaded model right after load/quant."""
        return self._model_vram_mb

    @staticmethod
    def _resolve_load_route(model_class: str, is_vlm: bool) -> str:
        """Map a model_class (+ is_vlm heuristic) to a loader route.

        Returns one of "vlm", "lm", "seq2seq".
        """
        explicit = {
            "causal_lm":          "lm",
            "image_text_to_text": "vlm",
            "vision2seq":         "vlm",
            "seq2seq_lm":         "seq2seq",
        }
        if model_class in explicit:
            return explicit[model_class]
        return "vlm" if is_vlm else "lm"     # model_class == "auto"

    def _apply_torchao(self, scheme: str) -> None:
        """Apply torchao on-the-fly weight quantization in place (HF runtime).

        A second quantization backend to compare against llm_compressor/modelopt and
        TurboQuant. Weight-only int8/int4 and dynamic int8; fp8 weight-only on Ada/
        Hopper. int4/fp8 need a kernel that may be unavailable on older torch/GPU —
        those raise and the benchmark records a load error (crash-safe).
        """
        from torchao.quantization import (
            quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
        )
        configs = {
            "int8wo": lambda: Int8WeightOnlyConfig(),
            "int8dq": lambda: Int8DynamicActivationInt8WeightConfig(),
            "int4wo": lambda: Int4WeightOnlyConfig(group_size=128),
        }
        try:
            from torchao.quantization import Float8WeightOnlyConfig
            configs["fp8wo"] = lambda: Float8WeightOnlyConfig()
        except Exception:
            pass
        if scheme not in configs:
            raise ValueError(f"Unknown torchao scheme '{scheme}'. Choose: {list(configs)}")
        quantize_(self.model, configs[scheme]())
        logger.info("[HFRuntime] torchao '%s' applied.", scheme)

    def _maybe_enable_turboquant(self, entry: "BenchmarkModelEntry") -> None:
        """Patch attention + arm a TurboQuant cache when enabled for this entry."""
        tq = getattr(entry, "turboquant", None)
        if tq is None or not getattr(tq, "enabled", False):
            return
        if self._is_vlm:
            logger.warning("[HFRuntime] TurboQuant on VLMs is not supported yet — "
                           "skipping for '%s'.", self._model_id)
            return
        from .config import CacheConfig
        from .patcher import patch_model_attention
        self._tq_cfg = CacheConfig(
            ring_capacity=tq.ring_capacity,
            key_bits=tq.key_bits,
            value_bits=tq.value_bits,
            use_qjl=tq.use_qjl,
            use_compressed_store=tq.use_compressed_store,
            device=str(self._device),
            dtype=(getattr(self.model, "dtype", None) and str(self.model.dtype).split(".")[-1]) or "bfloat16",
        )
        n_patched = patch_model_attention(self.model)
        if n_patched == 0:
            logger.warning("[HFRuntime] TurboQuant: no attention modules matched the "
                           "duck-type — TQ will not engage. Disabling.")
            self._tq_cfg = None
            return
        logger.info("[HFRuntime] TurboQuant enabled: %d attn layers patched | "
                    "ring=%d K%d/V%d use_qjl=%s store=%s",
                    n_patched, tq.ring_capacity, tq.key_bits, tq.value_bits,
                    tq.use_qjl, tq.use_compressed_store)

    def _sanitize_generation_config(self) -> None:
        """Drop sampling-only fields baked into the checkpoint's generation_config.

        Qwen/SmolVLM checkpoints ship temperature/top_p/top_k defaults. Our eval +
        latency paths run greedy (``do_sample=False``), so transformers emits a noisy
        "generation flags are not valid and may be ignored: ['temperature']" warning
        on every call. Clearing these here silences it. The sampling path in
        ``generate()`` (temperature > 0) re-supplies them explicitly with
        ``do_sample=True``, so sampling still works."""
        gc = getattr(self.model, "generation_config", None)
        if gc is None:
            return
        gc.do_sample = False
        for field in ("temperature", "top_p", "top_k"):
            if getattr(gc, field, None) is not None:
                setattr(gc, field, None)

    def _load_causal_lm(
        self,
        model_id:   str,
        quant_cfg,
        dtype:      torch.dtype,
        device_map: str,
        trust_rc:   bool,
        seq2seq:    bool = False,
    ) -> None:
        """Loads a decoder-only (or, when ``seq2seq``, encoder-decoder) LM + tokenizer."""
        if seq2seq:
            from transformers import AutoModelForSeq2SeqLM
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM

        try:
            logger.info(f"[HFRuntime] trying to load model normally...")
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
            # Encoder-decoder models pad on the right (encoder side).
            self.tokenizer.padding_side = "right" if seq2seq else "left"

            self.model = model_cls.from_pretrained(
                model_id, **_model_load_kwargs(quant_cfg, dtype, device_map, trust_rc),
            )
            self.model.eval()

        except:
            logger.info(f"[HFRuntime] Trying to load compressed model states...")
            compressed = _detect_compressed(model_id)

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

            if compressed:
                try:
                    import compressed_tensors  # noqa: F401 — activates hooks
                except ImportError:
                    raise ImportError(
                        "compressed-tensors is required for pack-quantized models.\n"
                        "Install: pip install llmcompressor"
                    )
            self.model = model_cls.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto"
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
        Loads a vision-language model via AutoModelForImageTextToText.

        This is the generic image-text-to-text class that resolves the correct
        architecture for modern VLMs — Qwen2.5-VL (Qwen2_5_VLForConditionalGeneration),
        LLaVA, SmolVLM, HunyuanOCR (custom, via trust_remote_code), etc. The older
        LlavaForConditionalGeneration path only worked for LLaVA-style models and
        produced wrong inputs for Qwen/Hunyuan.
        """
        _require_pil()

        # HunyuanOCR's processor needs use_fast=False; try fast first, fall back.
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_rc,
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_rc, use_fast=False,
            )
        self.tokenizer = self.processor.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **_model_load_kwargs(quant_cfg, dtype, device_map, trust_rc),
            )
        except Exception:
            logger.warning(
                f"[HFRuntime] AutoModelForImageTextToText failed for '{model_id}'. "
                "Falling back to AutoModelForCausalLM (trust_remote_code).",
                exc_info=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, **_model_load_kwargs(quant_cfg, dtype, device_map, True),
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
        self._tq_cfg   = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("[HFRuntime] Unload complete.")

    
    # Text generation
    

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
        gen_kwargs.update(self._tq_gen_kwargs(len(prompts)))   # TurboQuant when enabled (bs=1)

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

    
    # VLM generation
    def generate_vlm(
        self,
        image,
        prompt:         str,
        max_new_tokens: int,
    ) -> str:
        """
        Multimodal generation for modern VLMs (Qwen2.5-VL, LLaVA, SmolVLM, Hunyuan…).

        Builds a chat message with image + text content and renders it through the
        processor's chat template so the correct image-placeholder tokens are
        inserted. Raw ``processor(text=..., images=...)`` (the old path) does NOT
        insert those tokens for Qwen/Hunyuan and produces a token/feature mismatch.

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

        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        elif isinstance(image, PILImage.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            raise TypeError(
                f"image must be a PIL.Image or a file path string, "
                f"got {type(image).__name__}."
            )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }]

        # Preferred: chat template tokenizes + inserts image tokens in one call.
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt = True,
                tokenize              = True,
                return_dict           = True,
                return_tensors        = "pt",
            ).to(self._device)
        except Exception:
            # Fallback: render template to text, then call processor with images.
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt",
            ).to(self._device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                pad_token_id   = self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        )

    
    # Latency profiling
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
                    **self._tq_gen_kwargs(1),
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
                    min_new_tokens          = OUTPUT_TOKENS,   # force full length for stable TPOT
                    do_sample               = False,
                    return_dict_in_generate = True,
                    output_scores           = True,
                    pad_token_id            = self.tokenizer.pad_token_id,
                    **self._tq_gen_kwargs(1),
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

    
    # Throughput profiling
    

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
                        # TurboQuant cache only at bs=1 (_tq_gen_kwargs returns {} otherwise).
                        self.model.generate(
                            **inputs,
                            max_new_tokens = output_len,
                            min_new_tokens = output_len,   # force full length (no early EOS) for stable tok/s
                            do_sample      = False,
                            pad_token_id   = self.tokenizer.pad_token_id,
                            **self._tq_gen_kwargs(bs),
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
                # TurboQuant is single-sequence; bs>1 silently runs the default cache,
                # so flag whether this row actually exercised TQ (avoids misleading bars).
                "turboquant":     bool(self._tq_cfg is not None and bs == 1),
            })

            logger.info(
                f"[HFRuntime] bs={bs:>3} | "
                f"tps={tokens_per_sec:>8.1f} | "
                f"latency={median_latency_ms:>7.1f} ms"
            )

        return results

    
    # Memory
    def peak_vram_mb(self) -> float:
        """
        Returns peak VRAM allocated (in MB) since last reset.
        Returns 0.0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Max context probe
    def _model_max_positions(self) -> Optional[int]:
        """Best-effort read of the model's max position embeddings (handles VLM text_config)."""
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return None
        for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)):
            if obj is None:
                continue
            v = getattr(obj, "max_position_embeddings", None)
            if isinstance(v, int) and v > 0:
                return v
        return None

    def measure_max_context(
        self,
        start:     int = 1024,
        hard_cap:  int = 131072,
        run_forward: bool = True,
    ) -> dict:
        """
        Probe the largest prompt length the loaded model can process on this GPU.

        Doubles the sequence length from ``start`` until either a forward pass OOMs
        or the model's ``max_position_embeddings`` cap is reached, then binary-searches
        between the last success and first failure for the precise boundary.

        Args:
            start:       Initial sequence length to try.
            hard_cap:    Absolute ceiling regardless of model config (safety bound).
            run_forward: If True, runs an actual forward pass (real OOM boundary).
                         If False, only reports the model's configured max length.

        Returns:
            {
                "model_max_positions": int | None,   # config cap
                "measured_max_tokens": int | None,    # largest len that ran (None if not probed)
                "oom_at_tokens":       int | None,    # first len that OOM'd
                "capped_by":           "config" | "oom" | "hard_cap" | "config_only",
            }
        """
        self._check_loaded()
        model_max = self._model_max_positions()
        ceiling = min(hard_cap, model_max) if model_max else hard_cap

        if not run_forward or not torch.cuda.is_available():
            return {
                "model_max_positions": model_max,
                "measured_max_tokens": None,
                "oom_at_tokens":       None,
                "capped_by":           "config_only",
            }

        vocab = int(getattr(self.model.config, "vocab_size", 32000) or 32000)
        total_vram_mb = torch.cuda.get_device_properties(self._device).total_memory / (1024 ** 2)

        def _fits(length: int) -> bool:
            try:
                torch.cuda.reset_peak_memory_stats()
                ids = torch.randint(0, vocab, (1, length), device=self._device)
                with torch.no_grad():
                    self.model(input_ids=ids)
                peak_mb = self.peak_vram_mb()
                del ids
                torch.cuda.empty_cache()
                if peak_mb > total_vram_mb * _VRAM_OVERSUBSCRIBE_SAFETY:
                    # "Succeeded" only via WDDM shared-memory paging — not a real fit.
                    logger.warning(
                        "[max_context] length=%d peak=%.0fMB exceeds dedicated VRAM "
                        "(%.0fMB) — treating as not-fits (shared-memory fallback).",
                        length, peak_mb, total_vram_mb,
                    )
                    return False
                return True
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    return False
                raise

        last_ok: Optional[int] = None
        first_fail: Optional[int] = None
        length = start
        # Exponential growth phase.
        while length <= ceiling:
            if _fits(length):
                last_ok = length
                length *= 2
            else:
                first_fail = length
                break
        capped_by = "oom"
        if first_fail is None:
            # never OOM'd up to the ceiling
            capped_by = "config" if (model_max and ceiling == model_max) else "hard_cap"
            return {
                "model_max_positions": model_max,
                "measured_max_tokens": last_ok,
                "oom_at_tokens":       None,
                "capped_by":           capped_by,
            }

        # Binary-search between last_ok and first_fail for the precise boundary.
        lo = last_ok or start // 2
        hi = first_fail
        while hi - lo > max(256, lo // 16):
            mid = (lo + hi) // 2
            if _fits(mid):
                lo = mid
            else:
                hi = mid
        return {
            "model_max_positions": model_max,
            "measured_max_tokens": lo,
            "oom_at_tokens":       first_fail,
            "capped_by":           "oom",
        }

    # Context-length sweep: resident KV memory vs context length (TurboQuant-aware)
    def _kv_bytes_per_token(self) -> int:
        """Resident bytes of FP16 KV cache for ONE token, across all layers (K+V)."""
        cfg = self.model.config
        n_layers = getattr(cfg, "num_hidden_layers", 0)
        n_kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0))
        head_dim = getattr(cfg, "hidden_size", 0) // max(getattr(cfg, "num_attention_heads", 1), 1)
        return n_layers * n_kv * head_dim * 2 * 2   # 2 bytes (bf16) * 2 (K and V)

    def _tq_resident_kv_mb(self, cache) -> float:
        """Resident KV MB held by a TurboQuantCache (exact ring bf16 + packed store)."""
        total = 0
        for eng in cache.engines:
            ring = eng.ring.size * cache.num_kv_heads * cache.head_dim * 2 * 2  # bf16 K+V
            total += ring + eng.store.memory_bytes()
        return total / (1024 ** 2)

    def measure_ctx_sweep(self, lengths: list[int]) -> dict:
        """Resident KV-cache memory at increasing context lengths (the metric where
        TurboQuant actually shows up — KV dominates only at long context).

        For each length L it prefills L random tokens through the model with a cache
        (TurboQuant when enabled, else the default), then records the *resident* KV
        memory and peak VRAM. Stops at the first OOM. Also estimates the max context
        that fits = free-VRAM-after-load / KV-bytes-per-token (compressed for TQ).

        Returns:
            {
              "series": [{context_len, kv_cache_mb, peak_vram_mb, fits}, ...],
              "max_fit_tokens": int,            # last length that fit a forward
              "est_max_ctx_tokens": int | None, # free VRAM / per-token KV bytes
              "kv_bytes_per_token_fp16": int,
              "turboquant": bool,
            }
        """
        self._check_loaded()
        vocab = int(getattr(self.model.config, "vocab_size", 32000) or 32000)
        fp16_bpt = self._kv_bytes_per_token()
        series, max_fit = [], 0
        total_vram_mb = torch.cuda.get_device_properties(self._device).total_memory / (1024 ** 2)

        for L in sorted(lengths):
            if not torch.cuda.is_available():
                break
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                ids = torch.randint(0, vocab, (1, L), device=self._device)
                cache = self._maybe_tq_cache(1)   # TQ cache when enabled, else None
                with torch.no_grad():
                    if cache is not None:
                        self.model(input_ids=ids, past_key_values=cache, use_cache=True)
                        kv_mb = self._tq_resident_kv_mb(cache)
                    else:
                        self.model(input_ids=ids, use_cache=True)
                        kv_mb = (L * fp16_bpt) / (1024 ** 2)   # dense fp16 KV
                fp16_kv_mb = (L * fp16_bpt) / (1024 ** 2)
                peak_mb = round(self.peak_vram_mb(), 1)
                # WDDM can silently page an allocation into shared system RAM instead
                # of raising OOM — a peak above the card's physical VRAM means this
                # "success" ran at PCIe speed, not a real fit. Flag it and stop; don't
                # count it toward max_fit_tokens.
                oversubscribed = peak_mb > total_vram_mb * _VRAM_OVERSUBSCRIBE_SAFETY
                series.append({
                    "context_len": L,
                    "kv_cache_mb": round(kv_mb, 3),
                    "fp16_kv_mb": round(fp16_kv_mb, 3),
                    "compression_ratio": round(fp16_kv_mb / kv_mb, 2) if kv_mb else 1.0,
                    "peak_vram_mb": peak_mb,
                    "fits": True,
                    "oversubscribed": oversubscribed,
                })
                del ids, cache
                torch.cuda.empty_cache()
                if oversubscribed:
                    logger.warning(
                        "[ctx_sweep] context_len=%d peak=%.0fMB exceeds dedicated VRAM "
                        "(%.0fMB) — shared-memory fallback, not a real fit. Stopping sweep.",
                        L, peak_mb, total_vram_mb,
                    )
                    break
                max_fit = L
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise
                series.append({"context_len": L, "fits": False})
                torch.cuda.empty_cache()
                break

        # Estimate max context from free VRAM and the per-token KV cost. For TQ the
        # effective per-token cost is the measured compressed rate at the largest L.
        est_max = None
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            bpt = fp16_bpt
            tq_pts = [s for s in series if s.get("fits") and s["context_len"] > self._tq_cfg.ring_capacity] \
                if self._tq_cfg is not None else []
            if tq_pts:
                s = tq_pts[-1]
                bpt = max((s["kv_cache_mb"] * (1024 ** 2)) / s["context_len"], 1)
            est_max = int(free_b / bpt)

        return {
            "series": series,
            "max_fit_tokens": max_fit,
            "est_max_ctx_tokens": est_max,
            "kv_bytes_per_token_fp16": fp16_bpt,
            "turboquant": self._tq_cfg is not None,
        }

    # TurboQuant bits x accuracy x context-length characterization
    def _real_input_ids(self, max_len: int) -> "Tensor":
        """A real token sequence of length max_len (wikitext; random fallback)."""
        try:
            from src.utils import load_wikitext
            ds = load_wikitext(split="test", num_samples=200)
            text = "\n\n".join(t for t in ds["text"] if t and t.strip())
            ids = self.tokenizer(text, return_tensors="pt").input_ids[:, :max_len]
            if ids.shape[1] >= max_len:
                return ids.to(self._device)
        except Exception as e:
            logger.warning("[bits_sweep] wikitext load failed (%s); using random tokens.", e)
        vocab = int(getattr(self.model.config, "vocab_size", 32000) or 32000)
        return torch.randint(0, vocab, (1, max_len), device=self._device)

    def measure_bits_accuracy(
        self,
        bit_pairs: list[list[int]],
        lengths:   list[int],
        ring_capacity: int = 256,
    ) -> dict:
        """Next-token agreement vs FP16 KV, swept over bit budgets x context length.

        TurboQuant is a decode-time KV codec, so its quality impact only appears in
        the attention/decode path. For each context length L we take a real prompt,
        compute the FP16 next-token distribution, then for each [key_bits, value_bits]
        rebuild the model's KV through a TurboQuantCache and measure how much the
        predictions changed:
          - top1_agreement: fraction of positions whose argmax matches FP16
          - kl_div:         mean KL(FP16 || TQ) over the compared positions
        Positions before ``ring_capacity`` are exact (in the ring), so only the
        compressed tail [ring_capacity:] is compared.

        Returns {"grid": [{key_bits, value_bits, bits, context_len, top1_agreement,
        kl_div, n_positions}], "ring_capacity": int}.
        """
        self._check_loaded()
        if not torch.cuda.is_available():
            return {"skipped": "needs CUDA"}
        from .cache import TurboQuantCache
        from .config import CacheConfig
        from .patcher import patch_model_attention
        patch_model_attention(self.model)   # idempotent; defers to original for non-TQ caches

        dtype_name = (getattr(self.model, "dtype", None) and str(self.model.dtype).split(".")[-1]) or "bfloat16"
        grid = []
        max_len = max(lengths)
        full_ids = self._real_input_ids(max_len)

        for L in sorted(lengths):
            if L <= ring_capacity:
                continue   # nothing compressed → trivially identical to FP16
            ids = full_ids[:, :L]
            # Compare only the LAST `compare` positions. In real decode the ring always
            # holds the *current* token's most-recent `ring_capacity` tokens exact, with
            # only the far history compressed — that's the regime that matters. Comparing
            # all [ring:] positions is unrealistically harsh because mid-sequence tokens'
            # recent context lands in the compressed store, not the ring.
            compare = min(64, L - ring_capacity)
            try:
                with torch.no_grad():
                    base_logits = self.model(input_ids=ids, use_cache=False).logits[0]  # (L, V)
                base_arg = base_logits[-compare:].argmax(dim=-1)                          # (compare,)
                base_logp = torch.log_softmax(base_logits[-compare:].float(), dim=-1)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise
                torch.cuda.empty_cache()
                break   # bigger L will also OOM

            for kb, vb in bit_pairs:
                try:
                    cfg = CacheConfig(ring_capacity=ring_capacity, key_bits=kb, value_bits=vb,
                                      use_qjl=False, use_compressed_store=True,
                                      device=str(self._device), dtype=dtype_name)
                    cache = TurboQuantCache(self.model.config, cfg)
                    with torch.no_grad():
                        tq_logits = self.model(input_ids=ids, past_key_values=cache,
                                               use_cache=True).logits[0]
                    tq_slice = tq_logits[-compare:]
                    agree = (tq_slice.argmax(dim=-1) == base_arg).float().mean().item()
                    # Top-5 agreement: FP16's argmax lands in TQ's top-5 (argmax alone
                    # is too strict — a small prob shift between synonyms flips it).
                    tq_top5 = tq_slice.topk(5, dim=-1).indices
                    top5 = (tq_top5 == base_arg.unsqueeze(-1)).any(dim=-1).float().mean().item()
                    # KL(FP16 || TQ), averaged over positions. kl_div(input=log q,
                    # target=p) computes sum p*(log p - log q); base_logp is log p.
                    tq_logp = torch.log_softmax(tq_slice.float(), dim=-1)
                    kl = torch.nn.functional.kl_div(
                        tq_logp, base_logp, log_target=True, reduction="batchmean"
                    ).item()
                    kv_mb = self._tq_resident_kv_mb(cache)
                    fp16_kv_mb = (L * self._kv_bytes_per_token()) / (1024 ** 2)
                    grid.append({
                        "key_bits": kb, "value_bits": vb, "bits": f"K{kb}V{vb}",
                        "context_len": L,
                        "top1_agreement": round(agree, 4),
                        "top5_agreement": round(top5, 4),
                        "kl_div": round(kl, 5),
                        "kv_cache_mb": round(kv_mb, 3),
                        "compression_ratio": round(fp16_kv_mb / kv_mb, 2) if kv_mb else None,
                        "n_positions": int(base_arg.numel()),
                    })
                    del cache, tq_logits
                    torch.cuda.empty_cache()
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                        raise
                    torch.cuda.empty_cache()
            del base_logits, base_logp
            torch.cuda.empty_cache()

        return {"grid": grid, "ring_capacity": ring_capacity}

    # Extras: score_choices (used by eval_mmlu_tiny)
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
    
    # Internal guards
    def _check_loaded(self) -> None:
        """Raises RuntimeError if load() has not been called yet."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "HFRuntime: model is not loaded. Call load(entry) first."
            )
    
    # Dunder helpers
    def __repr__(self) -> str:
        status = f"loaded='{self._model_id}'" if self.model else "unloaded"
        return f"HFRuntime({status}, device={self._device}, vlm={self._is_vlm})"

    def __enter__(self) -> "HFRuntime":
        """Supports usage as a context manager."""
        return self

    def __exit__(self, *_) -> None:
        """Automatically unloads on context manager exit."""
        self.unload()