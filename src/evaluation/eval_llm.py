"""LLM text quality evaluation: perplexity, MMLU, token agreement, KL-divergence.

Provides sliding-window perplexity on calibration datasets (WikiText, etc.) and
multiple-choice accuracy on MMLU subjects. Also computes logit-based agreement and
KL-divergence between quantized and baseline models for detailed quality analysis.
"""

import math
import logging
from typing import Any, Optional
import torch
import torch.nn.functional as F
from src.utils import load_mmlu, load_wikitext

logger = logging.getLogger(__name__)


def _runtime_supports_logits(runtime) -> bool:
    """Check if runtime can return raw logits via forward_logits().

    Both HF and vLLM runtimes define forward_logits(), but vLLM raises
    NotImplementedError (the engine hides logits), so a bare hasattr() check
    is insufficient — gate on the runtime name instead.

    Args:
        runtime: Runtime instance to check.

    Returns:
        True if runtime supports logit access (HFRuntime); False for vLLM or if method absent.
    """
    if not hasattr(runtime, "forward_logits"):
        return False
    return getattr(runtime, "name", "") != "vllm"


def compute_ppl(
    runtime,
    dataset,
    ctx_len: int = 2048,
    stride: int = 512,
    text_key: str = "text",
    max_chunks: Optional[int] = None,
) -> float:
    """Compute perplexity using sliding-window cross-entropy on a dataset.

    Concatenates all text from the dataset, tokenizes it, and scores each token
    using a sliding window. Each window scores only the new tokens (overlap with
    previous window via stride), avoiding double-counting and enabling evaluation
    on sequences longer than the model context length.

    Args:
        runtime: Runtime with tokenizer and forward_logits() (HFRuntime only).
        dataset: HuggingFace Dataset with a text column.
        ctx_len: Context window length (e.g., 2048).
        stride: Sliding-window stride; must be < ctx_len (e.g., 512).
        text_key: Column name containing text data.
        max_chunks: Optional limit on number of chunks to process (for debugging).

    Returns:
        Perplexity (exp of mean negative log-likelihood).

    Raises:
        ValueError: If runtime does not support logits (vLLM) or no tokens scored.
    """
    if not _runtime_supports_logits(runtime):
        raise ValueError(
            "compute_ppl requires logit access via forward_logits(). "
            "The current runtime does not support it (vLLM hides logits) — "
            "use HFRuntime for perplexity evaluation."
        )

    full_text = "\n\n".join(
        row[text_key] for row in dataset if row[text_key].strip()
    )
    encodings = runtime.tokenizer(full_text, return_tensors="pt", truncation=False,)
    input_ids = encodings.input_ids
    total_len = input_ids.size(1)

    nlls = []
    token_count = 0
    chunk_idx = 0

    for begin in range(0, total_len, stride):
        end = min(begin + ctx_len, total_len)
        chunk_ids = input_ids[:, begin:end]
        chunk_len = chunk_ids.size(1)

        if begin == 0:
            score_start = 1
        else:
            score_start = max(1, chunk_len - stride)

        if score_start >= chunk_len:
            if end == total_len:
                break
            continue

        target_len = chunk_len - score_start
        loss_val = _nll_from_logits(runtime, chunk_ids, score_start, chunk_len)

        nlls.append(loss_val * target_len)
        token_count += target_len
        chunk_idx += 1

        if max_chunks and chunk_idx >= max_chunks:
            break
        if end == total_len:
            break

    if token_count == 0:
        raise ValueError("No tokens were scored — check dataset and tokenizer.")

    avg_nll = sum(nlls) / token_count
    return math.exp(avg_nll)

def _nll_from_logits(
    runtime,
    chunk_ids: torch.Tensor,
    score_start: int,
    chunk_len: int,
) -> float:
    """Compute mean negative log-likelihood for tokens [score_start:chunk_len].

    Args:
        runtime: Runtime instance with forward_logits method.
        chunk_ids: Input IDs tensor of shape (1, chunk_len).
        score_start: First token position to score (usually 1 or chunk_len - stride).
        chunk_len: Total chunk length.

    Returns:
        Mean cross-entropy loss for the scored tokens.
    """
    with torch.no_grad():
        logits = runtime.forward_logits(chunk_ids)

    shift_logits = logits[:, score_start - 1 : chunk_len - 1, :]
    shift_labels = chunk_ids[:, score_start : chunk_len]

    shift_logits = shift_logits.contiguous().float()
    shift_labels = shift_labels.contiguous().to(
        shift_logits.device
    ).long()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )

    return loss.item()

# 2.3.2  MMLU Tiny Evaluation
_CHOICE_LABELS = ["A", "B", "C", "D"]

_MMLU_TEMPLATE = (
    "{question}\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}\n"
    "Answer:"
)


def eval_mmlu_tiny(
    runtime,
    subject_list: list[str],
    num_q_per_subject: int = 100,
    seed: int = 42,
) -> float:
    """Evaluate LLM accuracy on MMLU multiple-choice questions.

    For each question, scores the log-probability of each answer choice (A, B, C, D)
    under the model, selects the highest-scoring choice, and compares to the gold label.

    Args:
        runtime: Runtime with forward_logits() (HFRuntime) or score_choices() method.
        subject_list: List of MMLU subjects (e.g., ['abstract_algebra', 'biology']).
        num_q_per_subject: Number of questions sampled per subject (total = len * num_q).
        seed: Random seed for reproducible subject/question sampling.

    Returns:
        Accuracy as a float (0.0-1.0).
        Accuracy in [0, 1]. Higher is better.
    """
    correct = 0
    total   = 0

    #Need progress bar with proper label
    for subject in subject_list:
        try:
            ds = load_mmlu(subject=subject, split="test",
                           num_samples=num_q_per_subject, seed=seed)
        except Exception as e:
            logger.warning(f"Could not load MMLU subject '{subject}': {e}")
            continue

        for row in ds:
            prompt  = _format_mmlu_prompt(row)
            gold_idx = int(row["answer"])          # 0-indexed int in MMLU
            gold_label = _CHOICE_LABELS[gold_idx]

            pred_label = _score_mmlu_choices(runtime, prompt)

            if pred_label == gold_label:
                correct += 1
            total += 1

    if total == 0:
        raise ValueError("No MMLU questions were evaluated.")

    return correct / total


def _format_mmlu_prompt(row: dict) -> str:
    """Formats one MMLU row into a multiple-choice prompt string."""
    choices = row["choices"]                        # list of 4 strings
    return _MMLU_TEMPLATE.format(
        question=row["question"],
        A=choices[0], B=choices[1],
        C=choices[2], D=choices[3],
    )


def _score_mmlu_choices(runtime, prompt: str) -> str:
    """
    Returns the answer label (A/B/C/D) with the highest log-probability
    given the prompt. Uses forward_logits if available.
    """
    if hasattr(runtime, "score_choices"):
        # Some runtimes expose a direct scoring method
        scores = runtime.score_choices(prompt, _CHOICE_LABELS)
        return max(scores, key=scores.get)

    # ── Token log-prob via forward_logits ────────────────────────────────────
    inputs  = runtime.tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = runtime.forward_logits(inputs.input_ids)  # (1, seq, vocab)

    last_logits = logits[0, -1, :]                         # distribution over next token
    log_probs   = F.log_softmax(last_logits, dim=-1)

    best_label, best_lp = None, float("-inf")
    for label in _CHOICE_LABELS:
        token_id = runtime.tokenizer.encode(
            f" {label}", add_special_tokens=False
        )[0]
        lp = log_probs[token_id].item()
        if lp > best_lp:
            best_lp, best_label = lp, label

    return best_label



# 2.3.3  Logit KL Divergence
def eval_logit_kl(
    runtime,
    baseline_logits_dict: dict[str, torch.Tensor],
    reduction: str = "mean",
) -> float:
    """
    Computes mean token-wise KL divergence between the runtime's logits
    and a set of pre-computed baseline logits.

    Args:
        runtime:               Runtime with forward_logits().
        baseline_logits_dict:  Dict mapping prompt strings → baseline logit
                               tensors of shape (1, seq_len, vocab_size).
        reduction:             'mean' averages KL over all tokens and prompts.

    Returns:
        Mean KL divergence (float). Lower means closer to baseline.

    Raises:
        AttributeError: If runtime lacks forward_logits().
    """
    if not hasattr(runtime, "forward_logits"):
        raise AttributeError(
            "eval_logit_kl requires runtime.forward_logits(). "
            "vLLM runtimes are not supported here."
        )

    kl_scores = []

    for prompt, base_logits in baseline_logits_dict.items():
        inputs = runtime.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            curr_logits = runtime.forward_logits(inputs.input_ids)

        seq_len = min(curr_logits.size(1), base_logits.size(1))

        curr_lp = F.log_softmax(curr_logits[:, :seq_len, :], dim=-1)  # (1, T, V)
        base_p  = F.softmax(base_logits[:, :seq_len, :],    dim=-1)   # (1, T, V)

        # KL(base || current) per token, then average
        kl_per_token = F.kl_div(
            curr_lp.reshape(-1, curr_lp.size(-1)),
            base_p.reshape(-1, base_p.size(-1)),
            reduction="batchmean",
        )
        kl_scores.append(kl_per_token.item())

    return sum(kl_scores) / len(kl_scores) if kl_scores else 0.0



# 2.3.4  Token Agreement
def eval_token_agreement(
    runtime,
    baseline_outputs: dict[str, str],
    max_new_tokens: int = 64,
) -> float:
    """
    Measures exact-match fraction between runtime greedy outputs
    and a set of baseline outputs on the same prompts.

    Args:
        runtime:          Runtime with generate().
        baseline_outputs: Dict mapping prompt strings → expected output strings.
        max_new_tokens:   Token budget for greedy generation.

    Returns:
        Exact-match fraction in [0, 1]. Higher means closer to baseline.
    """
    matches = 0
    total   = len(baseline_outputs)

    for prompt, expected in baseline_outputs.items():
        # runtime.generate takes a list of prompts and returns a list of strings;
        # temperature=0.0 selects greedy decoding.
        generated = runtime.generate(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        pred = generated[0].strip()

        if pred == expected.strip():
            matches += 1

    return matches / total if total > 0 else 0.0



# 2.3.5  Convenience: evaluate all metrics and pack into a result dict
def run_llm_eval(
    runtime,
    subject_list: list[str],
    baseline_logits_dict: Optional[dict]  = None,
    baseline_outputs:     Optional[dict]  = None,
    ppl_num_samples: int  = 256,
    num_q_per_subject: int = 100,
) -> dict:
    """
    Runs all LLM eval functions and returns a consolidated results dict.

    Returns:
        {
          "ppl":            float,
          "mmlu_acc":       float,
          "kl_div":         float | None,
          "token_agree":    float | None,
        }
    """
    results = {}

    # Perplexity
    wiki_ds = load_wikitext(split="test", num_samples=ppl_num_samples)
    results["ppl"] = compute_ppl(runtime, wiki_ds)

    # MMLU
    results["mmlu_acc"] = eval_mmlu_tiny(
        runtime, subject_list, num_q_per_subject=num_q_per_subject
    )

    # KL divergence (optional)
    if baseline_logits_dict:
        results["kl_div"] = eval_logit_kl(runtime, baseline_logits_dict)
    else:
        results["kl_div"] = None

    # Token agreement (optional)
    if baseline_outputs:
        results["token_agree"] = eval_token_agreement(runtime, baseline_outputs)
    else:
        results["token_agree"] = None

    return results