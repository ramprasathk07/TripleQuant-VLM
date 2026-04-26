"""
Perplexity evaluation using a HuggingFace model in eval mode.
"""
from __future__ import annotations

import logging
import math

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Calculates perplexity (PPL) of a model on a text dataset.

    Args:
        model: HuggingFace model with a ``forward()`` that accepts ``labels``.
        tokenizer: Tokenizer (or ``processor.tokenizer`` for VLMs).
        device: Torch device string.
    """

    def __init__(self, model, tokenizer=None, processor=None, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.device = device

    def calculate_ppl(self, texts: list[str], max_length: int = 2048) -> dict:
        """Compute perplexity over *texts*.

        Args:
            texts: List of plain text strings.
            max_length: Maximum token length per sample.

        Returns:
            Dictionary with ``perplexity`` and ``num_samples``.
        """
        self.model.eval()
        nlls: list[torch.Tensor] = []

        logger.info("Calculating perplexity over %d samples …", len(texts))
        for text in tqdm(texts, desc="PPL"):
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            input_ids = encodings.input_ids
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids.clone())
                nlls.append(outputs.loss)

        if not nlls:
            return {"perplexity": float("nan"), "num_samples": 0}

        ppl = math.exp(torch.stack(nlls).mean().item())
        logger.info("Perplexity: %.4f", ppl)
        return {"perplexity": ppl, "num_samples": len(nlls)}
