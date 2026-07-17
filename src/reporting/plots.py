"""Matplotlib chart generation for the report. Headless (Agg backend, no display).

Every function returns the written Path on success, or None if there wasn't
enough data to plot -- callers skip that section of the report rather than
embedding a broken image link.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .extract import bits_grid, ctx_rows

logger = logging.getLogger(__name__)

_NUM = (int, float)
# Consistent per-model color across every chart in a report (extend if a sweep
# ever compares more than 8 models at once).
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def _model_colors(models: list[str]) -> dict[str, str]:
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metric_bar(rows: list[dict], key: str, title: str, ylabel: str,
                    out_path: Path, lower_is_better: bool = True) -> Path | None:
    """Generic bar chart: one bar per model for a single scalar column."""
    candidates = [r for r in rows if r.get("status") == "success" and isinstance(r.get(key), _NUM)]
    if not candidates:
        logger.info("[report] skip %s: no rows with '%s'", out_path.name, key)
        return None

    colors = _model_colors([r["model"] for r in candidates])
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(candidates) + 2), 4))
    bars = ax.bar([r["model"] for r in candidates], [r[key] for r in candidates],
                  color=[colors[r["model"]] for r in candidates])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for bar, r in zip(bars, candidates):
        v = r[key]
        label = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    best = (min if lower_is_better else max)(candidates, key=lambda r: r[key])
    bars[candidates.index(best)].set_edgecolor("black")
    bars[candidates.index(best)].set_linewidth(2.0)
    return _save(fig, out_path)


def plot_ctx_sweep(detail_records: dict[str, dict], out_path: Path) -> Path | None:
    """Peak VRAM vs context length, one line per model that has ctx_sweep data.

    This is the hero chart: it's the only view where TurboQuant's actual value
    proposition (smaller KV cache => longer context on the same GPU) is visible
    at all -- short-context runs show fp16 and TurboQuant peak VRAM within noise
    of each other, since resident weights dominate until context grows.
    """
    series_by_model: dict[str, list[dict]] = {}
    for model, rec in detail_records.items():
        rows = ctx_rows(rec.get("metrics", {}))
        if rows:
            series_by_model[model] = sorted(rows, key=lambda s: s["context_len"])
    if not series_by_model:
        logger.info("[report] skip %s: no ctx_sweep data", out_path.name)
        return None

    colors = _model_colors(list(series_by_model))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, rows in series_by_model.items():
        ax.plot([s["context_len"] for s in rows], [s["peak_vram_mb"] for s in rows],
               marker="o", label=model, color=colors[model])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Peak VRAM vs context length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_tq_bits_tradeoff(detail_records: dict[str, dict], out_path: Path) -> Path | None:
    """TurboQuant's honest compression-vs-quality curve: next-token agreement vs
    fp16, per bit-width, averaged across the swept context lengths. Bars are
    annotated with the mean KV compression ratio at that bit-width.

    Kept even though low bit-widths score poorly here on purpose -- the point of
    this chart is to show the real tradeoff, not to make TurboQuant look better
    than the K3V2 default measures.
    """
    grid: list[dict] = []
    for rec in detail_records.values():
        grid.extend(bits_grid(rec.get("metrics", {})))
    if not grid:
        logger.info("[report] skip %s: no tq_bits_sweep data", out_path.name)
        return None

    by_bits: dict[str, list[dict]] = {}
    for row in grid:
        by_bits.setdefault(row["bits"], []).append(row)
    # Sort by total bit budget (key_bits + value_bits) so the x-axis reads low->high.
    labels = sorted(by_bits, key=lambda b: sum(by_bits[b][0][k] for k in ("key_bits", "value_bits")))

    top1_mean = [sum(r["top1_agreement"] for r in by_bits[b]) / len(by_bits[b]) for b in labels]
    comp_mean = [sum(r["compression_ratio"] for r in by_bits[b]) / len(by_bits[b]) for b in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, top1_mean, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 agreement vs FP16 (mean over context lengths)")
    ax.set_title("TurboQuant: quality vs bit-width (compression ratio annotated)")
    for bar, comp in zip(bars, comp_mean):
        ax.annotate(f"{comp:.2f}x KV", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_path)
