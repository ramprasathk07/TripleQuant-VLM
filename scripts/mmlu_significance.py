#!/usr/bin/env python
"""Paired significance test for MMLU accuracy differences between models.

Why this exists: comparing two models' MMLU accuracies as bare numbers invites
reading noise as signal. At n=250 the binomial standard error is ~3.1pp, so a
model scoring 0.556 vs a baseline's 0.548 (+0.8pp = 2 questions out of 250) looks
like an improvement and is nothing of the sort. The leaderboard genuinely had
"quantization improved MMLU" rows that were 2-3 flipped questions.

Every model in a sweep answers the *same* questions in the same order, so the
right tool is McNemar's exact test on the discordant pairs (questions where
exactly one of the two models is correct) -- far more sensitive than comparing
two independent proportions, because it conditions on the shared questions.

Usage:
    python scripts/mmlu_significance.py --dir results/qwen3-1.7b-leaderboard
    python scripts/mmlu_significance.py --dir results/... --baseline qwen3-1.7b-fp16-baseline
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _binom_two_sided_p(b: int, c: int) -> float:
    """McNemar exact (binomial) two-sided p-value for discordant counts b, c.

    Under H0 each discordant question is an independent coin flip, so the count
    of one direction is Binomial(b+c, 0.5). Exact rather than the chi-square
    approximation because discordant counts here are often small (<25).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) + P(X >= n-k) == 2 * P(X <= k) for the symmetric case.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _load_per_question(path: Path) -> tuple[str, list[dict]] | None:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    detail = rec.get("metrics", {}).get("quality_llm", {})
    if not isinstance(detail, dict):
        return None
    mmlu = detail.get("mmlu_detail")
    if not isinstance(mmlu, dict) or not mmlu.get("per_question"):
        return None
    return rec.get("model_name", path.stem), mmlu


def main() -> None:
    p = argparse.ArgumentParser(description="Paired McNemar test on MMLU results")
    p.add_argument("--dir", "-d", required=True, help="results/<run_name> directory")
    p.add_argument("--baseline", default=None,
                   help="model_name to compare against (default: the one containing 'fp16')")
    p.add_argument("--run-ts", default=None,
                   help="restrict to one run timestamp (default: the latest found)")
    args = p.parse_args()

    results_dir = Path(args.dir)
    files = [f for f in results_dir.glob("*.json") if not f.name.startswith("comparison_summary")]
    if args.run_ts:
        files = [f for f in files if args.run_ts in f.name]
    else:
        stamps = sorted({f.stem.split("_")[-1] for f in files})
        if stamps:
            files = [f for f in files if f.stem.endswith(stamps[-1])]

    models: dict[str, dict] = {}
    for f in files:
        loaded = _load_per_question(f)
        if loaded:
            models[loaded[0]] = loaded[1]

    if not models:
        print(f"No MMLU per-question data found in {results_dir}.\n"
              "Only sweeps run after the mmlu_detail change record it — re-run the benchmark.")
        sys.exit(1)

    base_name = args.baseline
    if base_name is None:
        base_name = next((m for m in models if "fp16" in m), None)
    if base_name not in models:
        print(f"Baseline '{base_name}' not among: {sorted(models)}")
        sys.exit(1)

    base = models[base_name]
    base_hits = [q["hit"] for q in base["per_question"]]
    n = base["total"]
    se = base["stderr"]

    print(f"MMLU paired significance vs baseline '{base_name}'")
    print(f"n = {n} questions | baseline acc = {base['acc']:.4f} "
          f"({base['correct']}/{n}) | unpaired SE = {se*100:.2f}pp "
          f"(95% CI +/-{1.96*se*100:.2f}pp)")
    print()
    hdr = f"{'model':<38} {'acc':>7} {'delta':>8} {'b':>4} {'c':>4} {'p':>8}  verdict"
    print(hdr)
    print("-" * len(hdr))

    for name, m in sorted(models.items()):
        if name == base_name:
            continue
        hits = [q["hit"] for q in m["per_question"]]
        if len(hits) != len(base_hits):
            print(f"{name:<38} SKIPPED (question count {len(hits)} != baseline {len(base_hits)})")
            continue
        # b: baseline right, model wrong (regressions). c: model right, baseline wrong.
        b = sum(1 for x, y in zip(base_hits, hits) if x and not y)
        c = sum(1 for x, y in zip(base_hits, hits) if y and not x)
        pval = _binom_two_sided_p(b, c)
        delta = m["acc"] - base["acc"]
        verdict = "SIGNIFICANT" if pval < 0.05 else "not significant (noise)"
        print(f"{name:<38} {m['acc']:>7.4f} {delta*100:>+7.1f}pp {b:>4} {c:>4} {pval:>8.4f}  {verdict}")

    print()
    print("b = baseline correct, model wrong.  c = model correct, baseline wrong.")
    print("p from McNemar's exact test on the b+c discordant questions (H0: quantization")
    print("is as likely to fix a question as to break one). p >= 0.05 means the accuracy")
    print("difference is indistinguishable from noise -- do NOT report it as a finding.")


if __name__ == "__main__":
    main()
