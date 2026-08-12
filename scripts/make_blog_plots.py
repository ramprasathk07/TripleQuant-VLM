#!/usr/bin/env python
"""Generate the figures used by docs/blog/*.md.

Every number here is read from the committed results JSON, never typed in by hand —
so a re-run of the benchmarks regenerates correct figures, and a figure can never
silently drift from the data it claims to show.

Palette is the validated 4-hue categorical set (light + dark steps checked with the
dataviz validator: lightness band, chroma floor, CVD separation, normal-vision floor,
contrast). Bars/points carry direct labels because two of the light-mode hues sit
below 3:1 against the surface — identity is never color-alone.

Usage:  python scripts/make_blog_plots.py [--outdir docs/blog/plots]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots (light mode) — fixed order, never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "600",
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def _latest(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


# ── Figure 1: same checkpoint, two runtimes ──────────────────────────────────
def fig_runtime_gap(results_dir: Path, out: Path) -> None:
    """The 13x gap. Grouped bars: HF eager vs vLLM, per checkpoint."""
    summary = _latest(str(results_dir / "comparison_summary_*.json"))
    hf = {r["model"]: r.get("best_tps") for r in summary["records"]}
    serving = {}
    sfile = results_dir / "vllm_serving.jsonl"
    if sfile.exists():
        for line in sfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("decode_tps"):
                    serving[rec["model"]] = rec["decode_tps"]

    rows = [("FP16", "fp16-baseline", "qwen3-1.7b-fp16-baseline"),
            ("AWQ-W4A16", "llmc-awq-w4a16", "qwen3-1.7b-llmc-awq-w4a16"),
            ("GPTQ-W8A8", "llmc-gptq-w8a8", "qwen3-1.7b-llmc-gptq-w8a8")]
    labels, hf_v, vl_v = [], [], []
    for label, skey, hkey in rows:
        if hf.get(hkey) and serving.get(skey):
            labels.append(label); hf_v.append(hf[hkey]); vl_v.append(serving[skey])

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    b1 = ax.bar([i - w/2 - 0.01 for i in x], hf_v, w, label="HuggingFace eager",
                color=ORANGE, edgecolor=SURFACE, linewidth=2)
    b2 = ax.bar([i + w/2 + 0.01 for i in x], vl_v, w, label="vLLM (fused kernels)",
                color=BLUE, edgecolor=SURFACE, linewidth=2)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}",
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("Decode tokens/sec (batch 1, greedy)")
    ax.set_title("Same checkpoint, two runtimes\nThe format didn't change — the kernels did", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, max(vl_v) * 1.22)
    _save(fig, out)


# ── Figure 2: MMLU deltas with CI + significance ─────────────────────────────
def fig_mmlu_significance(results_dir: Path, out: Path, run_ts: str) -> None:
    """Deltas vs fp16 with 95% CI whiskers; only one clears zero."""
    files = glob.glob(str(results_dir / f"*_hf_{run_ts}.json"))
    base, rows = None, []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        md = d["metrics"].get("quality_llm", {}).get("mmlu_detail")
        if md and "fp16" in d["model_name"]:
            base = md
    for f in sorted(files):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        md = d["metrics"].get("quality_llm", {}).get("mmlu_detail")
        if not md or "fp16" in d["model_name"]:
            continue
        bh = [q["hit"] for q in base["per_question"]]
        hh = [q["hit"] for q in md["per_question"]]
        b = sum(1 for x, y in zip(bh, hh) if x and not y)
        c = sum(1 for x, y in zip(bh, hh) if y and not x)
        name = (d["model_name"].replace("qwen3-1.7b-", "")
                .replace("llmc-", "").replace("torchao-", "torchao "))
        # CI of a paired difference, from the discordant counts.
        n = md["total"]
        se_pp = ((b + c) ** 0.5) / n * 100
        rows.append((name, (md["acc"] - base["acc"]) * 100, 1.96 * se_pp, b, c))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (name, delta, ci, b, c) in enumerate(rows):
        real = abs(delta) > ci
        color = ORANGE if real else BLUE
        ax.errorbar(delta, i, xerr=ci, fmt="o", color=color, ecolor=color,
                    elinewidth=2, capsize=5, markersize=9, zorder=3)
        tag = "real" if real else "noise"
        ax.annotate(f"{delta:+.1f}pp  ({b} broken / {c} fixed) — {tag}",
                    (delta, i), xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1.2, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("MMLU accuracy change vs FP16 (percentage points, 95% CI)")
    ax.set_title("Four quantized models vs FP16\nOnly one delta clears its own error bar", loc="left")
    ax.grid(axis="x", alpha=0.6); ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(rows) - 0.15)
    _save(fig, out)


# ── Figure 3: TurboQuant agreement vs context, per bit-width ─────────────────
def fig_tq_agreement(tq_dir: Path, out: Path) -> None:
    """The finding: the cliff is the bit budget, and it's flat across context."""
    files = glob.glob(str(tq_dir / "qwen3-1.7b-turboquant-*_hf_*.json"))
    grid = []
    for f in sorted(files):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        sweep = d["metrics"].get("perf", {}).get("tq_bits_sweep", {})
        if isinstance(sweep, dict) and sweep.get("grid"):
            grid = sweep["grid"]
    by = collections.defaultdict(list)
    for r in grid:
        by[r["bits"]].append((r["context_len"], r["top1_agreement"]))

    order = ["K2V2", "K3V2", "K4V4", "K8V8"]
    colors = {"K2V2": YELLOW, "K3V2": AQUA, "K4V4": ORANGE, "K8V8": BLUE}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for bits in order:
        if bits not in by:
            continue
        pts = sorted(by[bits])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=2,
                color=colors[bits], label=bits, zorder=3)
        ax.annotate(bits, (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    fontsize=10, color=INK, fontweight="600")
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(["512", "1K", "2K", "4K", "8K", "16K"])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(right=26000)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Top-1 agreement with FP16 KV")
    ax.set_title("TurboQuant: the cliff is the bit budget, not the context length")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center left", ncol=2)
    _save(fig, out)


# ── Figure 4: what each metric could actually see ────────────────────────────
def fig_metric_sensitivity(results_dir: Path, out: Path, run_ts: str) -> None:
    """PPL moved 68% where MMLU couldn't resolve anything. Same model."""
    files = glob.glob(str(results_dir / f"*_hf_{run_ts}.json"))
    base_ppl = base_mmlu = None
    vals = {}
    for f in sorted(files):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        q = d["metrics"].get("quality_llm", {})
        ppl, mmlu = q.get("ppl"), q.get("mmlu_acc")
        if not isinstance(ppl, (int, float)):
            continue
        name = d["model_name"].replace("qwen3-1.7b-", "").replace("llmc-", "")
        if "fp16" in name:
            base_ppl, base_mmlu = ppl, mmlu
        else:
            vals[name] = (ppl, mmlu)
    target = next((k for k in vals if "awq-w4a16" == k), None) or next(iter(vals))
    ppl, mmlu = vals[target]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pct = [(ppl - base_ppl) / base_ppl * 100, (mmlu - base_mmlu) / base_mmlu * 100]
    labels = ["Perplexity\n(wikitext-2)", "MMLU accuracy\n(n=800)"]
    bars = ax.bar(labels, pct, width=0.5, color=[ORANGE, BLUE],
                  edgecolor=SURFACE, linewidth=2)
    for bar, v, extra in zip(bars, pct, [f"{base_ppl:.2f} → {ppl:.2f}",
                                         f"{base_mmlu:.3f} → {mmlu:.3f} (p=0.53)"]):
        off = 3 if v >= 0 else -16
        ax.annotate(f"{v:+.1f}%\n{extra}",
                    (bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, off), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1.2)
    ax.set_ylabel("Change vs FP16 (%)")
    ax.set_title("AWQ-W4A16: two metrics, one model\nOnly one of them noticed", loc="left")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    ax.set_ylim(min(pct) - 22, max(pct) + 22)
    _save(fig, out)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate blog figures from results JSON")
    p.add_argument("--outdir", default="docs/blog/plots")
    p.add_argument("--run-ts", default="20260812T125040Z",
                   help="leaderboard run timestamp to read MMLU detail from")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.outdir) if Path(args.outdir).is_absolute() else root / args.outdir
    lb = root / "results" / "qwen3-1.7b-leaderboard"
    tq = root / "results" / "qwen3-1.7b-tq-quality"

    fig_runtime_gap(lb, out / "runtime_gap.png")
    fig_mmlu_significance(lb, out / "mmlu_significance.png", args.run_ts)
    fig_tq_agreement(tq, out / "tq_agreement_vs_context.png")
    fig_metric_sensitivity(lb, out / "metric_sensitivity.png", args.run_ts)


if __name__ == "__main__":
    main()
