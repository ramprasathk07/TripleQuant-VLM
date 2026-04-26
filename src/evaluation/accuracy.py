"""
Accuracy evaluation using a vLLM engine.

Computes a simple sequence-similarity score (1 − WER) per sample,
aggregated as mean accuracy over the evaluation set.
"""
from __future__ import annotations

import difflib
import logging

from datasets import load_dataset
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate — lower is better, 0 is perfect."""
    ref_words = str(reference).lower().split()
    hyp_words = str(hypothesis).lower().split()
    if not ref_words:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, ref_words, hyp_words).ratio()


class AccuracyEvaluator:
    """Evaluates text accuracy (via WER) on a HuggingFace dataset.

    Args:
        llm_engine: A pre-initialised ``vllm.LLM`` instance.
    """

    def __init__(self, llm_engine: LLM) -> None:
        self.llm = llm_engine

    def eval_ocr(
        self,
        dataset_id: str,
        split: str = "test",
        num_samples: int = 50,
        prompts: list[str] | None = None,
        max_new_tokens: int = 256,
    ) -> dict:
        """Evaluate OCR accuracy on a HuggingFace dataset.

        The evaluator attempts to read ground-truth text from the common field
        names used by OCR benchmarks (``"text"``, ``"ground_truth"``).

        Args:
            dataset_id: HuggingFace dataset repository id.
            split: Dataset split to use.
            num_samples: Number of samples to evaluate.
            prompts: List of user prompts to rotate over. Defaults to a
                standard OCR transcription prompt.
            max_new_tokens: Maximum tokens to generate per sample.

        Returns:
            Dictionary with ``avg_accuracy``, ``avg_wer``, and ``num_samples``.
        """
        if prompts is None:
            prompts = ["Read and transcribe the text in this image."]

        logger.info("Loading eval dataset: %s (%s split, %d samples)", dataset_id, split, num_samples)
        ds = load_dataset(dataset_id, split=split, streaming=True)

        scores: list[float] = []
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            prompt = prompts[i % len(prompts)]

            # vLLM multi-modal call (Qwen2-VL style)
            generate_input = {"prompt": prompt}
            if "image" in item:
                generate_input["multi_modal_data"] = {"image": item["image"]}

            try:
                outputs = self.llm.generate([generate_input], sampling_params, use_tqdm=False)
                prediction = outputs[0].outputs[0].text
            except Exception as exc:
                logger.warning("Generation failed for sample %d: %s", i, exc)
                scores.append(0.0)
                continue

            ref = item.get("text", item.get("ground_truth", ""))
            wer = calculate_wer(str(ref), prediction)
            scores.append(1.0 - wer)

        avg_acc = sum(scores) / len(scores) if scores else 0.0
        logger.info("Accuracy evaluation done: avg_accuracy=%.4f over %d samples", avg_acc, len(scores))
        return {
            "avg_accuracy": avg_acc,
            "avg_wer": 1.0 - avg_acc,
            "num_samples": len(scores),
        }
