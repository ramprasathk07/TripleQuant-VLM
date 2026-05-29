# eval_ocr.py  (continued from above)
"""
2.4 – OCR Evaluation  (VLMs only)
CER, WER, Exact Match, BLEU, per-sample breakdown.
"""

import logging
from typing import Optional
from src.utils import load_latex_ocr, load_hf_dataset, build_prompts, normalize_latex

logger = logging.getLogger(__name__)

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    logger.warning("jiwer not installed — CER/WER will be unavailable. "
                   "Install with: pip install jiwer")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not installed — BLEU will be unavailable. "
                   "Install with: pip install nltk")


# ════════════════════════════════════════════════════════════════════════════════
# 2.4.1  Text Normalization for OCR Comparison
# ════════════════════════════════════════════════════════════════════════════════

def _normalize_for_ocr(text: str) -> str:
    """
    Prepares a LaTeX string for metric computation:
      - Applies normalize_latex() from utils
      - Lowercases for case-insensitive comparison
      - Strips enclosing math-mode delimiters ($, $$, \\[ \\])
    """
    text = normalize_latex(text)
    for wrapper in (r"\[", r"\]", "$$", "$"):
        text = text.replace(wrapper, "")
    return text.strip().lower()


# ════════════════════════════════════════════════════════════════════════════════
# 2.4.2  Individual Metric Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate via jiwer (treats each char as a 'word')."""
    if not JIWER_AVAILABLE:
        return float("nan")
    h_chars = " ".join(list(hypothesis))
    r_chars = " ".join(list(reference))
    try:
        return jiwer.cer(r_chars, h_chars)
    except Exception:
        return float("nan")


def _compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate via jiwer."""
    if not JIWER_AVAILABLE:
        return float("nan")
    try:
        return jiwer.wer(reference, hypothesis)
    except Exception:
        return float("nan")


def _compute_bleu(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU (character-tokenized) via nltk."""
    if not NLTK_AVAILABLE:
        return float("nan")
    smoothie  = SmoothingFunction().method4
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars or not hyp_chars:
        return 0.0
    try:
        return sentence_bleu([ref_chars], hyp_chars, smoothing_function=smoothie)
    except Exception:
        return float("nan")


def _compute_exact_match(hypothesis: str, reference: str) -> bool:
    """Exact string match after normalization."""
    return hypothesis.strip() == reference.strip()


# ════════════════════════════════════════════════════════════════════════════════
# 2.4.3  Main OCR Evaluation Function
# ════════════════════════════════════════════════════════════════════════════════

_DEFAULT_OCR_PROMPT = (
    "You are an expert at reading mathematical notation.\n"
    "Look at the following image of a mathematical formula and transcribe it "
    "exactly into LaTeX.\n"
    "Respond with only the LaTeX expression, nothing else."
)

# Known OCR datasets → column mapping + recommended prompt. Auto-detect covers
# the rest, so this is just a fast path / override.
# NOTE: "config" is the HF dataset config/subset name. linxy/LaTeX_OCR is a
# multi-config hub dataset — load_dataset REQUIRES a config name, and only the
# "full"/"small"/"synthetic_handwrite" configs expose train/validation/test
# (the "default" config is train-only). Omitting it (the old behaviour) raised
# ValueError at load → all OCR evals hard-failed.
OCR_DATASET_REGISTRY: dict[str, dict] = {
    "linxy/LaTeX_OCR":    {"image": "image", "text": "text",         "split": "test", "config": "full"},
    "unsloth/LaTeX_OCR":  {"image": "image", "text": "text",         "split": "test"},
    "getomni-ai/ocr-benchmark": {"image": "image", "text": "ground_truth", "split": "test"},
}

# Candidate column names for auto-detection when a field is not given/known.
_IMAGE_CANDIDATES = ("image", "img", "pixel_values", "image_path")
_TEXT_CANDIDATES  = ("text", "formula", "latex", "label", "caption",
                     "answer", "ground_truth", "markdown", "transcription")


def _resolve_ocr_fields(
    columns: list[str],
    dataset_name: str,
    image_field: Optional[str],
    text_field: Optional[str],
) -> tuple[str, str]:
    """Resolve (image_col, text_col) from explicit args → registry → auto-detect."""
    reg = OCR_DATASET_REGISTRY.get(dataset_name, {})
    img = image_field or reg.get("image")
    txt = text_field  or reg.get("text")

    if img not in columns:
        img = next((c for c in _IMAGE_CANDIDATES if c in columns), None)
    if txt not in columns:
        txt = next((c for c in _TEXT_CANDIDATES if c in columns), None)

    if img is None or txt is None:
        raise ValueError(
            f"Could not resolve OCR image/text columns for '{dataset_name}'. "
            f"Available: {columns}. Set datasets.ocr_image_field / ocr_text_field."
        )
    logger.info("OCR fields resolved: image='%s' text='%s'", img, txt)
    return img, txt


def eval_ocr(
    runtime,
    dataset_name:    str           = "linxy/LaTeX_OCR",
    num_samples:     int           = 100,
    max_new_tokens:  int           = 256,
    split:           Optional[str] = None,
    seed:            int           = 42,
    prompt_template: Optional[str] = None,
    image_field:     Optional[str] = None,
    text_field:      Optional[str] = None,
    subset:          Optional[str] = None,
) -> dict:
    """
    Evaluates a VLM runtime on an OCR dataset (image → LaTeX).

    Args:
        runtime:         VLM runtime exposing generate_vlm(image, prompt, max_new_tokens).
        dataset_name:    HuggingFace dataset identifier. Default: 'linxy/LaTeX_OCR'.
        num_samples:     Number of samples to evaluate.
        max_new_tokens:  Max tokens the VLM may generate per sample.
        split:           Dataset split ('train' or 'test').
        seed:            Random seed for reproducible sampling.
        prompt_template: Optional override for the instruction prompt string.

    Returns:
        {
            "cer":          float,   # Character Error Rate  (↓ better)
            "wer":          float,   # Word Error Rate       (↓ better)
            "exact_match":  float,   # Exact-match fraction  (↑ better)
            "bleu":         float,   # Mean sentence BLEU    (↑ better)
            "num_samples":  int,     # Actual samples scored
            "per_sample":   list[dict],
        }

    Raises:
        AttributeError: If runtime does not expose generate_vlm().
        ValueError:     If no samples could be evaluated.

    Example:
        >>> results = eval_ocr(my_vlm_runtime, num_samples=50)
        >>> print(f"CER: {results['cer']:.4f} | BLEU: {results['bleu']:.4f}")
    """
    # ── Validate runtime ─────────────────────────────────────────────────────
    if not hasattr(runtime, "generate_vlm"):
        raise AttributeError(
            "eval_ocr requires runtime.generate_vlm(image, prompt, max_new_tokens). "
            "Ensure you are passing a VLM-capable runtime, not a text-only one."
        )

    prompt = prompt_template or _DEFAULT_OCR_PROMPT

    reg = OCR_DATASET_REGISTRY.get(dataset_name, {})
    # Resolve split (explicit → registry → "test").
    if split is None:
        split = reg.get("split", "test")
    # Resolve config/subset (explicit → registry → None). Multi-config hub
    # datasets (e.g. linxy/LaTeX_OCR) require this or load_dataset raises.
    if subset is None:
        subset = reg.get("config")

    # ── Load dataset ─────────────────────────────────────────────────────────
    # Oversample (×2) so we still hit num_samples if some rows are corrupt.
    logger.info("Loading OCR dataset '%s' | config=%s | split=%s | n=%d",
                dataset_name, subset, split, num_samples)
    dataset = load_hf_dataset(dataset_name, subset=subset, split=split,
                              num_samples=num_samples * 2, seed=seed)

    # ── Resolve image / text columns (explicit → registry → auto-detect) ───────
    img_key, txt_key = _resolve_ocr_fields(
        dataset.column_names, dataset_name, image_field, text_field,
    )

    # ── Build prompts ─────────────────────────────────────────────────────────
    prompt_dicts = build_prompts(
        dataset,
        num_samples=num_samples,
        image_key=img_key,
        formula_key=txt_key,
        normalize=True,
        seed=seed,
    )

    # ── Inference + metrics ──────────────────────────────────────────────────
    per_sample   = []
    cer_scores   = []
    wer_scores   = []
    bleu_scores  = []
    exact_scores = []

    for i, item in enumerate(prompt_dicts):
        image        = item["image"]
        ground_truth = item["ground_truth"]   # already normalized by build_prompts
        ds_index     = item["index"]

        # ── Generate ─────────────────────────────────────────────────────────
        try:
            raw_pred = runtime.generate_vlm(
                image          = image,
                prompt         = prompt,
                max_new_tokens = max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"[sample {i} / ds_idx {ds_index}] generate_vlm failed: {e}")
            raw_pred = ""

        # ── Normalize prediction ──────────────────────────────────────────────
        prediction = _normalize_for_ocr(raw_pred)
        reference  = _normalize_for_ocr(ground_truth)

        # ── Compute per-sample metrics ────────────────────────────────────────
        cer   = _compute_cer(prediction, reference)
        wer   = _compute_wer(prediction, reference)
        bleu  = _compute_bleu(prediction, reference)
        exact = _compute_exact_match(prediction, reference)

        cer_scores.append(cer)
        wer_scores.append(wer)
        bleu_scores.append(bleu)
        exact_scores.append(float(exact))

        per_sample.append({
            "index":        ds_index,
            "prediction":   prediction,
            "ground_truth": reference,
            "cer":          cer,
            "wer":          wer,
            "bleu":         bleu,
            "exact_match":  exact,
        })

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i + 1}/{len(prompt_dicts)}] "
                f"CER={cer:.3f}  WER={wer:.3f}  "
                f"BLEU={bleu:.3f}  EM={exact}"
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(per_sample)
    if n == 0:
        raise ValueError("No OCR samples were successfully evaluated.")

    def _safe_mean(scores: list) -> float:
        """Mean that ignores NaN entries."""
        valid = [s for s in scores if s == s]   # NaN != NaN
        return sum(valid) / len(valid) if valid else float("nan")

    results = {
        "cer":         _safe_mean(cer_scores),
        "wer":         _safe_mean(wer_scores),
        "exact_match": _safe_mean(exact_scores),
        "bleu":        _safe_mean(bleu_scores),
        "num_samples": n,
        "per_sample":  per_sample,
    }

    # ── Summary log ──────────────────────────────────────────────────────────
    logger.info(
        f"\n{'─'*50}\n"
        f"OCR Eval complete  ({n} samples)\n"
        f"  CER          : {results['cer']:.4f}\n"
        f"  WER          : {results['wer']:.4f}\n"
        f"  Exact Match  : {results['exact_match']:.4f}\n"
        f"  BLEU         : {results['bleu']:.4f}\n"
        f"{'─'*50}"
    )

    return results


# ════════════════════════════════════════════════════════════════════════════════
# 2.4.4  Per-sample Report Helper
# ════════════════════════════════════════════════════════════════════════════════

def report_ocr_failures(
    results: dict,
    top_n:   int  = 10,
    metric:  str  = "cer",
) -> list[dict]:
    """
    Returns the top-N worst-performing samples ranked by a given metric.

    Args:
        results: Output dict from eval_ocr().
        top_n:   How many failure cases to return.
        metric:  Metric to rank by ('cer', 'wer', or 'bleu').
                 For 'bleu', ranking is ascending (lowest BLEU = worst).

    Returns:
        List of per-sample dicts, sorted worst-first.

    Example:
        >>> failures = report_ocr_failures(results, top_n=5, metric="cer")
        >>> for f in failures:
        ...     print(f["cer"], f["prediction"], f["ground_truth"])
    """
    per_sample = results.get("per_sample", [])
    if not per_sample:
        return []

    reverse = metric != "bleu"   # high CER/WER = bad; low BLEU = bad
    ranked  = sorted(
        per_sample,
        key    = lambda x: x.get(metric, 0.0),
        reverse= reverse,
    )
    return ranked[:top_n]


# ════════════════════════════════════════════════════════════════════════════════
# 2.4.5  Export Helpers
# ════════════════════════════════════════════════════════════════════════════════

def save_ocr_results(results: dict, path: str) -> None:
    """
    Saves the full eval_ocr() results dict to a JSON file.

    Args:
        results: Output dict from eval_ocr().
        path:    Destination file path (e.g. 'ocr_results.json').

    Note:
        PIL Image objects in per_sample[*]['image'] are stripped before saving
        since they are not JSON-serialisable.
    """
    import json

    serialisable = {
        k: v for k, v in results.items() if k != "per_sample"
    }
    serialisable["per_sample"] = [
        {sk: sv for sk, sv in sample.items() if sk != "image"}
        for sample in results.get("per_sample", [])
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)

    logger.info(f"OCR results saved → {path}")