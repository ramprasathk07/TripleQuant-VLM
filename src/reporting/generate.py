"""Orchestrates: load results -> compute verdicts -> render plots -> write report.md."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .loader import ReportData, load_report_data
from .markdown import render_markdown
from .plots import plot_ctx_sweep, plot_metric_bar, plot_tq_bits_tradeoff
from .verdicts import compute_verdicts

logger = logging.getLogger(__name__)


def _build_plots(data: ReportData, plots_dir: Path) -> dict[str, str]:
    """Generate every chart that has data; returns {name: path-relative-to-report-md}."""
    paths: dict[str, str] = {}

    p = plot_metric_bar(data.summary_rows, "tpot_ms_p50", "Time per output token",
                        "TPOT (ms, lower is better)", plots_dir / "latency.png", lower_is_better=True)
    if p:
        paths["latency"] = f"plots/{p.name}"

    p = plot_metric_bar(data.summary_rows, "peak_vram_mb", "Peak VRAM",
                        "VRAM (MB, lower is better)", plots_dir / "memory.png", lower_is_better=True)
    if p:
        paths["memory"] = f"plots/{p.name}"

    if any(isinstance(r.get("mmlu_acc"), (int, float)) for r in data.summary_rows):
        p = plot_metric_bar(data.summary_rows, "mmlu_acc", "MMLU-tiny accuracy",
                            "Accuracy (higher is better)", plots_dir / "quality.png", lower_is_better=False)
    else:
        p = plot_metric_bar(data.summary_rows, "ppl", "Perplexity",
                            "PPL (lower is better)", plots_dir / "quality.png", lower_is_better=True)
    if p:
        paths["quality"] = f"plots/{p.name}"

    p = plot_ctx_sweep(data.detail_records, plots_dir / "ctx_sweep.png")
    if p:
        paths["ctx_sweep"] = f"plots/{p.name}"

    p = plot_tq_bits_tradeoff(data.detail_records, plots_dir / "tq_bits_tradeoff.png")
    if p:
        paths["tq_bits"] = f"plots/{p.name}"

    return paths


def generate_report(results_dir: Path, summary_path: Path | None = None,
                    out_dir: Path | None = None) -> Path:
    """Turn a results/<run_name>/ directory into report.md + plots/ + summary.json.

    Returns the path to the written report.md.
    """
    results_dir = Path(results_dir)
    data = load_report_data(results_dir, summary_path)
    out_dir = Path(out_dir) if out_dir else (results_dir / "report")
    plots_dir = out_dir / "plots"

    logger.info("Loaded %d summary row(s), %d detail record(s) from %s",
               len(data.summary_rows), len(data.detail_records), data.summary_path)

    verdicts = compute_verdicts(data.summary_rows)
    plot_paths = _build_plots(data, plots_dir)

    report_md = render_markdown(data, verdicts, plot_paths)
    report_path = out_dir / "report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    summary_json = {
        "run_name": data.run_name,
        "timestamp": data.timestamp,
        "environment": data.environment,
        "source_summary": str(data.summary_path),
        "verdicts": verdicts,
        "records": data.summary_rows,
        "plots": plot_paths,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")

    return report_path
