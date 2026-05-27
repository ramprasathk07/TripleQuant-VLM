# eval_llm.py
"""
2.3 – LLM Text Evaluation
Perplexity, MMLU, KL-divergence, and token-agreement metrics.
"""

import math
import logging
from typing import Any, Optional
import torch
import torch.nn.functional as F
from src.utils import load_mmlu, load_wikitext

logger = logging.getLogger(__name__)


def _runtime_supports_logits(runtime) -> bool:
    """True if the runtime can return raw logits via forward_logits().

    Both HF and vLLM runtimes define forward_logits(), but vLLM raises
    NotImplementedError (the engine hides logits), so a bare hasattr() check
    is insufficient — gate on the runtime name instead.
    """
    if not hasattr(runtime, "forward_logits"):
        return False
    return getattr(runtime, "name", "") != "vllm"


# ════════════════════════════════════════════════════════════════════════════════
# 2.3.1  Perplexity  (sliding-window)
# ════════════════════════════════════════════════════════════════════════════════

def compute_ppl(
    runtime,
    dataset,
    ctx_len: int = 2048,
    stride:  int = 512,
    text_key: str = "text",
    max_chunks: Optional[int] = None,
) -> float:
    """
    Computes sliding-window perplexity over a text dataset.

    Requires a logit-capable runtime (HFRuntime). vLLM hides logits, so PPL
    is not supported there — raises ValueError telling the caller to switch.

    Args:
        runtime:    Runtime object exposing forward_logits().
        dataset:    HF Dataset with a text column.
        ctx_len:    Maximum context window length in tokens.
        stride:     Sliding-window stride in tokens.
        text_key:   Column name for raw text.
        max_chunks: Optional cap on total chunks processed (for speed).

    Returns:
        Perplexity score (float). Lower is better.

    Raises:
        ValueError: If the runtime cannot expose logits (e.g. vLLM).
    """
    if not _runtime_supports_logits(runtime):
        raise ValueError(
            "compute_ppl requires logit access via forward_logits(). "
            "The current runtime does not support it (vLLM hides logits) — "
            "use HFRuntime for perplexity evaluation."
        )

    # ── Concatenate all text into one token stream ───────────────────────────
    full_text = "\n\n".join(
        row[text_key] for row in dataset if row[text_key].strip()
    )
    encodings = runtime.tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids  # (1, total_len)
    total_len  = input_ids.size(1)

    nlls        = []
    token_count = 0
    chunk_idx   = 0

    for begin in range(0, total_len, stride):
        end         = min(begin + ctx_len, total_len)
        chunk_ids   = input_ids[:, begin:end]            # (1, chunk_len)
        target_len  = end - (begin + stride if begin > 0 else begin)

        if target_len <= 0:
            break

        # Tokens we actually score (the non-overlapping suffix)
        score_start = chunk_ids.size(1) - target_len

        nll = _nll_from_logits(runtime, chunk_ids, score_start)

        nlls.append(nll * target_len)
        token_count += target_len
        chunk_idx   += 1

        if max_chunks and chunk_idx >= max_chunks:
            break

        if end == total_len:
            break

    if token_count == 0:
        raise ValueError("No tokens were scored — check dataset and tokenizer.")

    avg_nll = sum(nlls) / token_count
    return math.exp(avg_nll)


def _nll_from_logits(runtime, chunk_ids: torch.Tensor, score_start: int) -> float:
    """Cross-entropy loss over the scoreable suffix, using forward_logits."""
    with torch.no_grad():
        logits = runtime.forward_logits(chunk_ids)   # (1, seq, vocab)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, score_start - 1 : -1, :]  # (1, target_len, vocab)
    shift_labels = chunk_ids[:, score_start:]           # (1, target_len)

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    return loss.item()


# ════════════════════════════════════════════════════════════════════════════════
# 2.3.2  MMLU Tiny Evaluation
# ════════════════════════════════════════════════════════════════════════════════

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
    """
    Evaluates LLM accuracy on MMLU multiple-choice questions.
    Scoring is done via log-probability of each answer letter token.

    Args:
        runtime:           Runtime with forward_logits() or score_choices().
        subject_list:      List of MMLU subjects (e.g. ['abstract_algebra']).
        num_q_per_subject: Number of questions sampled per subject.
        seed:              Sampling seed for reproducibility.

    Returns:
        Accuracy in [0, 1]. Higher is better.
    """
    correct = 0
    total   = 0

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


# ════════════════════════════════════════════════════════════════════════════════
# 2.3.3  Logit KL Divergence
# ════════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════════
# 2.3.4  Token Agreement
# ════════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════════
# 2.3.5  Convenience: evaluate all metrics and pack into a result dict
# ════════════════════════════════════════════════════════════════════════════════

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