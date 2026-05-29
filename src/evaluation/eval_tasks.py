"""Additional LLM benchmark evaluators: GSM8K, AIME, C-Eval, HumanEval.

These complement the perplexity / MMLU metrics in ``eval_llm.py``:

  - GSM8K     — grade-school math word problems; generative, numeric-answer match.
  - AIME      — competition math (integer 0-999 answers); generative, exact match.
  - C-Eval    — Chinese MMLU-style multiple choice; logit-scored A/B/C/D.
  - HumanEval — Python code generation; pass@1, OPTIONAL sandboxed execution.

All generative evals format the prompt through the model's chat template when one
is available (important for instruct / "thinking" models like Qwen3-4B-Thinking).

SECURITY: HumanEval can execute model-generated Python to compute pass@1. That is
disabled by default (``execute=False``) — it only generates + saves completions.
Enable execution only in a trusted/sandboxed environment.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import torch
import torch.nn.functional as F

from src.utils import load_hf_dataset

logger = logging.getLogger(__name__)

_CHOICE_LABELS = ["A", "B", "C", "D"]


# ════════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════════

def _format_chat(runtime, user_text: str, system: Optional[str] = None) -> str:
    """Apply tokenizer's chat template to user text with optional system prompt.

    Renders messages through the tokenizer's apply_chat_template() if available
    (e.g., for instruct/chat models). Falls back to raw text for base models or
    on any template application error.

    Args:
        runtime: Runtime instance with optional tokenizer attribute.
        user_text: User message content.
        system: Optional system message content.

    Returns:
        Formatted chat prompt (template-rendered if available, else raw user_text).
    """
    tok = getattr(runtime, "tokenizer", None)
    if tok is None or getattr(tok, "chat_template", None) is None:
        return user_text
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return user_text


def _generate_one(runtime, prompt: str, max_new_tokens: int) -> str:
    """Run greedy generation on a single prompt and return the completion.

    Args:
        runtime: Runtime instance with generate() method.
        prompt: Input prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text (decoded, no prompt echo); empty string on failure.
    """
    out = runtime.generate([prompt], max_new_tokens=max_new_tokens, temperature=0.0)
    return out[0] if out else ""


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text: str) -> Optional[float]:
    """Extract the last numeric value from text, handling thousands separators.

    Args:
        text: String to search.

    Returns:
        Last number found as float, or None if no number present.
    """
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _numbers_close(a: Optional[float], b: Optional[float], tol: float = 1e-4) -> bool:
    """Check if two numbers are close within relative tolerance.

    Args:
        a: First number.
        b: Second number (reference for relative tolerance).
        tol: Relative tolerance factor (default 1e-4).

    Returns:
        True if |a - b| <= tol * max(1.0, |b|); False if either is None.
    """
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def _score_mc(runtime, prompt: str, labels: list[str]) -> str:
    """Select highest log-probability choice label for multiple-choice question.

    Tries each label with and without leading space to handle both Latin prompts
    (MMLU-style "Answer: A") and Chinese prompts (C-Eval-style "答案：A"), selecting
    the variant with higher log-probability. Prefers runtime.score_choices() if available.

    Args:
        runtime: Runtime with tokenizer and forward_logits() (or score_choices()).
        prompt: Question + answer choices prompt.
        labels: List of choice labels, typically ['A', 'B', 'C', 'D'].

    Returns:
        Highest-scoring label.
    """
    if hasattr(runtime, "score_choices"):
        try:
            scores = runtime.score_choices(prompt, labels)
            return max(scores, key=scores.get)
        except Exception:
            pass

    tok = runtime.tokenizer
    inputs = tok(prompt, return_tensors="pt")
    logits = runtime.forward_logits(inputs.input_ids)
    log_probs = F.log_softmax(logits[0, -1, :].float(), dim=-1)

    best_label, best_lp = labels[0], float("-inf")
    for label in labels:
        lp = float("-inf")
        for variant in (f" {label}", label):
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                lp = max(lp, log_probs[ids[0]].item())
        if lp > best_lp:
            best_lp, best_label = lp, label
    return best_label


# ════════════════════════════════════════════════════════════════════════════════
# GSM8K — grade-school math (generative, numeric match)
# ════════════════════════════════════════════════════════════════════════════════

def eval_gsm8k(
    runtime,
    num_samples: int = 200,
    max_new_tokens: int = 512,
    seed: int = 42,
    dataset_name: str = "openai/gsm8k",
) -> dict:
    """Accuracy on GSM8K via final-number match.

    Returns ``{"acc": float, "n": int}``.
    """
    ds = load_hf_dataset(dataset_name, subset="main", split="test",
                         num_samples=num_samples, seed=seed)
    correct = 0
    total = 0
    for row in ds:
        gold_txt = row["answer"].split("####")[-1].strip()
        gold = _last_number(gold_txt)
        user = (
            "Solve the math problem. Show your reasoning, then give the final "
            "answer on a new line as 'The answer is <number>'.\n\n"
            f"Problem: {row['question']}"
        )
        pred_txt = _generate_one(runtime, _format_chat(runtime, user), max_new_tokens)
        if _numbers_close(_last_number(pred_txt), gold):
            correct += 1
        total += 1
    acc = correct / total if total else float("nan")
    logger.info("GSM8K: %.4f (%d/%d)", acc, correct, total)
    return {"acc": acc, "n": total}


# ════════════════════════════════════════════════════════════════════════════════
# AIME — competition math (generative, integer match)
# ════════════════════════════════════════════════════════════════════════════════

def eval_aime(
    runtime,
    num_samples: int = 30,
    max_new_tokens: int = 1024,
    seed: int = 42,
    dataset_name: str = "Maxwell-Jia/AIME_2024",
    split: str = "train",
) -> dict:
    """Accuracy on AIME via integer-answer match (answers are integers 0-999)."""
    ds = load_hf_dataset(dataset_name, split=split, num_samples=num_samples, seed=seed)
    # Resolve column names defensively across AIME dataset variants.
    cols = set(ds.column_names)
    q_key = next((k for k in ("Problem", "problem", "question") if k in cols), None)
    a_key = next((k for k in ("Answer", "answer", "solution") if k in cols), None)
    if q_key is None or a_key is None:
        raise ValueError(f"AIME dataset columns not recognized: {ds.column_names}")

    correct = 0
    total = 0
    for row in ds:
        gold = _last_number(str(row[a_key]))
        user = (
            "Solve the AIME problem. Reason step by step, then give the final "
            "integer answer on a new line as 'The answer is <integer>'.\n\n"
            f"Problem: {row[q_key]}"
        )
        pred_txt = _generate_one(runtime, _format_chat(runtime, user), max_new_tokens)
        pred = _last_number(pred_txt)
        if pred is not None and gold is not None and int(round(pred)) == int(round(gold)):
            correct += 1
        total += 1
    acc = correct / total if total else float("nan")
    logger.info("AIME: %.4f (%d/%d)", acc, correct, total)
    return {"acc": acc, "n": total}


# ════════════════════════════════════════════════════════════════════════════════
# C-Eval — Chinese multiple choice (logit-scored)
# ════════════════════════════════════════════════════════════════════════════════

_CEVAL_DEFAULT_SUBJECTS = [
    "computer_network", "high_school_mathematics", "logic",
    "college_programming", "marxism",
]

_CEVAL_TEMPLATE = (
    "以下是一道关于{subject}的单项选择题，请选出正确答案的选项。\n"
    "{question}\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n答案："
)


def eval_ceval(
    runtime,
    subjects: Optional[list[str]] = None,
    num_q_per_subject: int = 20,
    seed: int = 42,
    dataset_name: str = "ceval/ceval-exam",
    split: str = "val",
) -> dict:
    """Accuracy on C-Eval (val split has gold answers). Logit-scored A/B/C/D."""
    subjects = subjects or _CEVAL_DEFAULT_SUBJECTS
    correct = 0
    total = 0
    for subject in subjects:
        try:
            ds = load_hf_dataset(dataset_name, subset=subject, split=split,
                                 num_samples=num_q_per_subject, seed=seed)
        except Exception as e:
            logger.warning("C-Eval subject '%s' failed to load: %s", subject, e)
            continue
        for row in ds:
            prompt = _CEVAL_TEMPLATE.format(
                subject=subject, question=row["question"],
                A=row["A"], B=row["B"], C=row["C"], D=row["D"],
            )
            pred = _score_mc(runtime, prompt, _CHOICE_LABELS)
            if pred == str(row["answer"]).strip().upper():
                correct += 1
            total += 1
    if total == 0:
        raise ValueError("No C-Eval questions were evaluated.")
    acc = correct / total
    logger.info("C-Eval: %.4f (%d/%d)", acc, correct, total)
    return {"acc": acc, "n": total}


# ════════════════════════════════════════════════════════════════════════════════
# HumanEval — code generation (pass@1, optional sandboxed execution)
# ════════════════════════════════════════════════════════════════════════════════

def _run_humaneval_case(program: str, timeout: float) -> bool:
    """Execute one HumanEval program in a subprocess; True if it exits cleanly.

    SECURITY: runs model-generated Python. Only call with execute=True in a
    trusted/sandboxed environment.
    """
    import subprocess
    import sys
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(program)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, timeout=timeout,
        )
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def eval_humaneval(
    runtime,
    num_samples: int = 20,
    max_new_tokens: int = 512,
    seed: int = 42,
    execute: bool = False,
    timeout: float = 10.0,
    dataset_name: str = "openai/openai_humaneval",
) -> dict:
    """HumanEval pass@1.

    With ``execute=False`` (default) only generates completions and reports how
    many were produced (``executed=False``). With ``execute=True`` it appends each
    problem's unit tests and runs them in a subprocess to compute pass@1.
    """
    ds = load_hf_dataset(dataset_name, split="test", num_samples=num_samples, seed=seed)
    generated = 0
    passed = 0
    for row in ds:
        header = row["prompt"]                       # function signature + docstring
        completion = _generate_one(runtime, header, max_new_tokens)
        generated += 1
        if not execute:
            continue
        program = (
            header + completion + "\n\n"
            + row["test"] + "\n\n"
            + f"check({row['entry_point']})\n"
        )
        if _run_humaneval_case(program, timeout):
            passed += 1

    if not execute:
        logger.info("HumanEval: generated %d completions (execution disabled)", generated)
        return {"generated": generated, "executed": False}
    pass_at_1 = passed / generated if generated else float("nan")
    logger.info("HumanEval pass@1: %.4f (%d/%d)", pass_at_1, passed, generated)
    return {"pass@1": pass_at_1, "n": generated, "executed": True}
