import re
import random
from typing import Optional, Union

# ── Optional imports (graceful fallback if datasets not installed) ────────────
try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

   
# ════════════════════════════════════════════════════════════════════════════════
# 2.2.1  LaTeX Normalization
# ════════════════════════════════════════════════════════════════════════════════

# Commands that can be safely removed (display/formatting only)
_STRIP_COMMANDS = (
    r"\\displaystyle",
    r"\\textstyle",
    r"\\scriptstyle",
    r"\\scriptscriptstyle",
    r"\\left",
    r"\\right",
    r"\\,",
    r"\\;",
    r"\\:",
    r"\\!",
    r"\\quad",
    r"\\qquad",
    r"\\medspace",
    r"\\thinspace",
    r"\\thickspace",
)

_STRIP_PATTERN = re.compile("|".join(_STRIP_COMMANDS))
_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
_NEWLINES      = re.compile(r"\n+")


def normalize_latex(text: str) -> str:
    """
    Normalizes a LaTeX string by:
      1. Stripping leading/trailing whitespace.
      2. Removing display/spacing-only commands (\\displaystyle, \\left, etc.).
      3. Collapsing multiple spaces/tabs into a single space.
      4. Collapsing newlines into a single space.
      5. Normalizing common equivalent forms (e.g. \\frac vs inline).

    Args:
        text: Raw LaTeX string.

    Returns:
        Cleaned, normalized LaTeX string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    # Step 1 – strip outer whitespace
    text = text.strip()

    # Step 2 – remove pure formatting/spacing commands
    text = _STRIP_PATTERN.sub("", text)

    # Step 3 – collapse newlines
    text = _NEWLINES.sub(" ", text)

    # Step 4 – collapse multiple spaces
    text = _MULTI_SPACE.sub(" ", text)

    # Step 5 – normalize curly-brace spacing: "{ " → "{", " }" → "}"
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)

    # Step 6 – final strip
    text = text.strip()

    return text


# ════════════════════════════════════════════════════════════════════════════════
# 2.2.2  HuggingFace Dataset Loaders
# ════════════════════════════════════════════════════════════════════════════════

def _check_hf() -> None:
    """Raises ImportError if `datasets` is not installed."""
    if not HF_AVAILABLE:
        raise ImportError(
            "The `datasets` package is required. Install it with:\n"
            "  pip install datasets"
        )


def load_hf_dataset(
    name: str,
    subset: Optional[str] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
    seed: int = 42,
    streaming: bool = False,
) -> "Dataset":
    """
    Generic HuggingFace dataset loader with optional sampling.

    Args:
        name:        HF dataset name, e.g. 'wikitext', 'cais/mmlu', 'linxy/LaTeX_OCR'.
        subset:      Dataset config/subset name (e.g. 'wikitext-2-raw-v1', 'all').
        split:       Dataset split to load ('train', 'validation', 'test').
        num_samples: If set, randomly samples this many rows.
        seed:        Random seed for reproducible sampling.
        streaming:   If True, loads in streaming mode (no sampling supported).

    Returns:
        A HuggingFace Dataset (or IterableDataset if streaming=True).
    """
    _check_hf()

    dataset = load_dataset(name, subset, split=split, streaming=streaming)

    if not streaming and num_samples is not None:
        total = len(dataset)
        num_samples = min(num_samples, total)
        indices = random.Random(seed).sample(range(total), num_samples)
        dataset = dataset.select(indices)

    return dataset


# ── Specialised convenience loaders ─────────────────────────────────────────

def load_wikitext(
    version: str = "wikitext-2-raw-v1",
    split: str = "test",
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> "Dataset":
    """
    Loads WikiText (language-modelling benchmark).

    Args:
        version:     'wikitext-2-raw-v1' or 'wikitext-103-raw-v1'.
        split:       'train', 'validation', or 'test'.
        num_samples: Optional row limit.
        seed:        Sampling seed.

    Returns:
        Dataset with column 'text'.

    Example:
        >>> ds = load_wikitext(num_samples=500)
        >>> ds[0]["text"]
        ' = Valkyria Chronicles III = ...'
    """
    return load_hf_dataset(
        name="wikitext",
        subset=version,
        split=split,
        num_samples=num_samples,
        seed=seed,
    )


def load_mmlu(
    subject: str = "all",
    split: str = "test",
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> "Dataset":
    """
    Loads the MMLU (Massive Multitask Language Understanding) benchmark.

    Args:
        subject:     MMLU subject/subset (e.g. 'all', 'abstract_algebra',
                     'high_school_mathematics').
        split:       'test', 'validation', or 'dev'.
        num_samples: Optional row limit.
        seed:        Sampling seed.

    Returns:
        Dataset with columns: 'question', 'choices', 'answer'.

    Example:
        >>> ds = load_mmlu(subject="high_school_mathematics", num_samples=100)
        >>> ds[0]["question"]
        'What is the derivative of ...'
    """
    return load_hf_dataset(
        name="cais/mmlu",
        subset=subject,
        split=split,
        num_samples=num_samples,
        seed=seed,
    )


def load_latex_ocr(
    split: str = "test",
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> "Dataset":
    """
    Loads the LaTeX_OCR dataset (image → LaTeX formula pairs).

    Args:
        split:       'train' or 'test'.
        num_samples: Optional row limit.
        seed:        Sampling seed.

    Returns:
        Dataset with columns: 'image', 'text'.

    Note:
        linxy/LaTeX_OCR is multi-config; the 'full' config is used (it exposes
        train/validation/test — the 'default' config is train-only).

    Example:
        >>> ds = load_latex_ocr(num_samples=50)
        >>> ds[0]["text"]
        '\\frac{a}{b} + c^{2}'
    """
    return load_hf_dataset(
        name="linxy/LaTeX_OCR",
        subset="full",
        split=split,
        num_samples=num_samples,
        seed=seed,
    )


# ════════════════════════════════════════════════════════════════════════════════
# 2.2.3  Prompt Builder (OCR Tasks)
# ════════════════════════════════════════════════════════════════════════════════

# Default prompt template for LaTeX OCR evaluation
_DEFAULT_OCR_TEMPLATE = (
    "You are an expert at reading mathematical notation.\n"
    "Look at the following image of a mathematical formula and transcribe it "
    "exactly into LaTeX.\n"
    "Respond with only the LaTeX expression, nothing else.\n\n"
    "Formula image: {image}"
)


def build_prompts(
    dataset: "Dataset",
    num_samples: int,
    template: str = _DEFAULT_OCR_TEMPLATE,
    image_key: str = "image",
    formula_key: str = "formula",
    seed: int = 42,
    normalize: bool = True,
) -> list[dict]:
    """
    Builds a list of prompt dicts for OCR/LaTeX evaluation tasks.

    Each returned dict contains:
        - 'prompt'         : formatted instruction string
        - 'image'          : raw image object (PIL or path)
        - 'ground_truth'   : (optionally normalized) LaTeX ground-truth string
        - 'index'          : original dataset index

    Args:
        dataset:     A HuggingFace Dataset (e.g. from load_latex_ocr()).
        num_samples: Number of samples to build prompts for.
        template:    Prompt template string. Must contain '{image}' placeholder.
        image_key:   Column name for the image field.
        formula_key: Column name for the ground-truth LaTeX field.
        seed:        Random seed for reproducible sampling.
        normalize:   If True, applies normalize_latex() to ground-truth strings.

    Returns:
        List of dicts, one per sample.

    Raises:
        ValueError: If template lacks '{image}' placeholder.
        KeyError:   If expected columns are missing from the dataset.

    Example:
        >>> ds = load_latex_ocr(num_samples=200)
        >>> prompts = build_prompts(ds, num_samples=50)
        >>> prompts[0].keys()
        dict_keys(['prompt', 'image', 'ground_truth', 'index'])
    """
    # ── Validate template ────────────────────────────────────────────────────
    if "{image}" not in template:
        raise ValueError("Prompt template must contain an '{image}' placeholder.")

    # ── Validate dataset columns ─────────────────────────────────────────────
    missing = {image_key, formula_key} - set(dataset.column_names)
    if missing:
        raise KeyError(
            f"Dataset is missing expected columns: {missing}. "
            f"Available columns: {dataset.column_names}"
        )

    # ── Sample indices ───────────────────────────────────────────────────────
    total      = len(dataset)
    num_samples = min(num_samples, total)
    indices    = random.Random(seed).sample(range(total), num_samples)

    # ── Build prompt dicts ───────────────────────────────────────────────────
    prompts = []
    for idx in indices:
        row   = dataset[idx]
        image = row[image_key]
        gt    = row[formula_key]

        if normalize:
            gt = normalize_latex(gt)

        prompt_text = template.format(image="<image>")   # placeholder for multimodal models

        prompts.append({
            "prompt":       prompt_text,
            "image":        image,
            "ground_truth": gt,
            "index":        idx,
        })

    return prompts