"""
Dataset utilities for TripleQuant-VLM.

Provides helpers for building calibration datasets (quantization) and
preparing single-image inputs (benchmarking / interactive use).
"""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset


def get_vlm_dataset(
    dataset_id: str,
    processor,
    split: str = "train",
    num_samples: int = 64,
    prompt: str = "Transcribe the text in this image.",
    seed: int = 42,
):
    """Load and format a HuggingFace dataset for VLM calibration.

    The returned dataset has exactly two columns that llmcompressor
    expects: ``text`` (chat-template formatted prompt) and ``image``
    (PIL Image).

    Args:
        dataset_id: HuggingFace dataset repository id (e.g. ``nielsr/funsd``).
        processor: The model's AutoProcessor (used to apply the chat template).
        split: Dataset split to use.
        num_samples: Maximum number of samples to select.
        prompt: The user instruction injected into every sample.
        seed: Shuffle seed for reproducibility.

    Returns:
        A HuggingFace Dataset with ``text`` and ``image`` columns.
    """
    ds = load_dataset(dataset_id, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    def _transform(example):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"text": text, "image": example["image"]}

    return ds.map(_transform, remove_columns=ds.column_names)


def get_llm_dataset(
    dataset_id: str,
    tokenizer,
    dataset_subset: Optional[str] = None,
    split: str = "train",
    num_samples: int = 64,
    text_column: str = "text",
    seed: int = 42,
):
    """Load and format a HuggingFace dataset for LLM calibration.

    Args:
        dataset_id: HuggingFace dataset repository id (e.g. ``ptb_text_only``).
        tokenizer: The model's AutoTokenizer.
        split: Dataset split to use.
        num_samples: Maximum number of samples to select.
        text_column: Name of the text column in the dataset.
        seed: Shuffle seed for reproducibility.

    Returns:
        A HuggingFace Dataset with a ``text`` column.
    """
    if dataset_subset:
        ds = load_dataset(dataset_id, dataset_subset, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    if "sentence" in ds.column_names and text_column not in ds.column_names:
        text_column = "sentence"
        
    # Some datasets don't have enough samples if we don't filter out empty lines
    ds = ds.filter(lambda x: len(x[text_column].strip()) > 0)
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    def _transform(example):
        messages = [
            {"role": "user", "content": example[text_column]}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"text": text}

    return ds.map(_transform, remove_columns=ds.column_names)


def get_vlm_input(
    processor,
    prompt: str,
    image=None,
    device: str = "cuda",
) -> dict:
    """Prepare a single model input dict from a text prompt + optional PIL image.

    Args:
        processor: The model's AutoProcessor.
        prompt: Plain text user prompt.
        image: Optional PIL Image. If ``None``, a text-only message is built.
        device: Torch device string to move tensors to.

    Returns:
        A dict of tensors suitable for ``model.generate(**inputs, ...)``.
    """
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")

    return {k: v.to(device) for k, v in inputs.items()}
