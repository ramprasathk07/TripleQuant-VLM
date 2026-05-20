#!/usr/bin/env python
"""
benchmark.py — TripleQuant-VLM multi-model benchmark entry point.

Loads each model in the config sequentially, runs the enabled metrics,
saves per-model JSON immediately (crash-safe), then writes a final
comparison summary.

Usage:
    python benchmark.py --config configs/benchmark/ocr_comparison.yaml
    python benchmark.py --config configs/benchmark/ocr_comparison.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark")

from src.config import load_benchmark_config
from src.config.schemas import BenchmarkConfig, BenchmarkModelEntry, MetricsConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Convert a display name into a safe filename stem."""
    return name.lower().replace(" ", "_").replace("/", "-")


def _run_metrics_for_model(
    model_entry: BenchmarkModelEntry,
    config: BenchmarkConfig,
) -> dict:
    """Initialise vLLM for *model_entry* and run all enabled metrics.

    Returns a dict with metric results. On partial failure the failed metric
    key will map to ``{"error": "<traceback>"}``.
    """
    from src.evaluation.latency import VLLMLatencyProfiler
    from src.evaluation.accuracy import AccuracyEvaluator
    from src.evaluation.memory import MemoryProfiler, get_vram_usage

    mc: MetricsConfig = config.metrics
    ds = config.dataset

    results: dict = {}

    # Initialise vLLM engine (shared across metrics for this model)
    logger.info("Initialising vLLM engine for '%s' …", model_entry.name)
    profiler = VLLMLatencyProfiler(
        model_path=model_entry.path,
        gpu_memory_utilization=model_entry.gpu_memory_utilization,
        max_model_len=model_entry.max_model_len,
    )

    # ── Latency ──────────────────────────────────────────────────────────────
    if mc.latency:
        logger.info("[%s] Running latency benchmark …", model_entry.name)
        try:
            prompt = ds.prompts[0] if ds.prompts else "Describe this image."
            results["latency"] = profiler.measure_latency(
                prompt=prompt,
                max_new_tokens=mc.max_new_tokens,
                warmup_runs=mc.warmup_runs,
                timed_runs=mc.timed_runs,
            )
        except Exception:
            logger.warning("[%s] Latency benchmark failed.", model_entry.name, exc_info=True)
            results["latency"] = {"error": traceback.format_exc()}

    # ── Memory ───────────────────────────────────────────────────────────────
    if mc.memory:
        logger.info("[%s] Profiling memory …", model_entry.name)
        try:
            mem_profiler = MemoryProfiler()
            prompt = ds.prompts[0] if ds.prompts else "Describe this image."

            from vllm import SamplingParams
            sp = SamplingParams(max_tokens=mc.max_new_tokens, temperature=0)

            mem_stats = mem_profiler.profile(
                profiler.llm.generate,
                [prompt],
                sp,
                use_tqdm=False,
            )
            results["memory"] = mem_stats
        except Exception:
            logger.warning("[%s] Memory profiling failed.", model_entry.name, exc_info=True)
            results["memory"] = {"error": traceback.format_exc()}

    # ── Accuracy ─────────────────────────────────────────────────────────────
    if mc.accuracy:
        logger.info("[%s] Running accuracy evaluation on %s …", model_entry.name, ds.dataset_id)
        try:
            evaluator = AccuracyEvaluator(profiler.llm)
            results["accuracy"] = evaluator.eval_ocr(
                dataset_id=ds.dataset_id,
                split=ds.dataset_split,
                num_samples=ds.num_samples,
                prompts=ds.prompts,
                max_new_tokens=mc.max_new_tokens,
            )
        except Exception:
            logger.warning("[%s] Accuracy evaluation failed.", model_entry.name, exc_info=True)
            results["accuracy"] = {"error": traceback.format_exc()}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TripleQuant-VLM — Benchmark multiple models sequentially.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a benchmark YAML config (e.g. configs/benchmark/ocr_comparison.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print the plan without loading any model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load + validate config ────────────────────────────────────────────
    logger.info("Loading benchmark config: %s", args.config)
    try:
        config = load_benchmark_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("─" * 60)
    logger.info("Benchmark plan:")
    for i, m in enumerate(config.models, 1):
        logger.info("  [%d] %-20s → %s", i, m.name, m.path)
    logger.info("  Dataset  : %s  [%d samples]",
                config.dataset.dataset_id, config.dataset.num_samples)
    logger.info("  Metrics  : latency=%s  accuracy=%s  memory=%s  ppl=%s",
                config.metrics.latency, config.metrics.accuracy,
                config.metrics.memory, config.metrics.perplexity)
    logger.info("  Results  : %s/", config.results_dir)
    logger.info("─" * 60)

    if args.dry_run:
        logger.info("--dry-run: stopping before model load. Config is valid ✓")
        return

    # ── 2. Run each model sequentially ───────────────────────────────────────
    all_results: dict[str, dict] = {}
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for model_entry in config.models:
        logger.info("=" * 60)
        logger.info("▶  Starting model: %s", model_entry.name)
        logger.info("=" * 60)

        try:
            model_metrics = _run_metrics_for_model(model_entry, config)
            status = "success"
        except Exception:
            tb = traceback.format_exc()
            logger.error("[%s] Fatal error:\n%s", model_entry.name, tb)
            model_metrics = {"fatal_error": tb}
            status = "failed"

        record = {
            "model_name": model_entry.name,
            "model_path": model_entry.path,
            "status": status,
            "timestamp": run_ts,
            "metrics": model_metrics,
        }
        all_results[model_entry.name] = record

        # ── Save per-model result immediately (crash-safe) ──────────────────
        safe_name = _safe_filename(model_entry.name)
        per_model_path = results_dir / f"{safe_name}_{run_ts}.json"
        with per_model_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        logger.info("💾 Saved results → %s", per_model_path)

    # ── 3. Write comparison summary ───────────────────────────────────────────
    summary = _build_comparison_summary(all_results)
    summary_path = results_dir / f"comparison_summary_{run_ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("✅ Benchmark complete.")
    logger.info("   Summary → %s", summary_path)
    logger.info("=" * 60)
    _print_summary_table(summary)


def _build_comparison_summary(all_results: dict) -> dict:
    """Build a flat side-by-side comparison dict from per-model results."""
    rows = []
    for name, record in all_results.items():
        metrics = record.get("metrics", {})
        row: dict = {"model": name, "status": record.get("status", "unknown")}

        # Latency
        lat = metrics.get("latency", {})
        row["avg_ttft_ms"] = lat.get("avg_ttft_ms")
        row["avg_throughput_tps"] = lat.get("avg_throughput_tps")

        # Accuracy
        acc = metrics.get("accuracy", {})
        row["avg_accuracy"] = acc.get("avg_accuracy")
        row["avg_wer"] = acc.get("avg_wer")

        # Memory
        mem = metrics.get("memory", {})
        row["peak_vram_gb"] = mem.get("peak_vram_gb")

        rows.append(row)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_models": len(rows),
        "models": rows,
    }


def _print_summary_table(summary: dict) -> None:
    """Pretty-print the comparison table to stdout."""
    rows = summary.get("models", [])
    if not rows:
        return

    header = f"{'Model':<22} {'Status':<10} {'TTFT(ms)':<12} {'TPS':<10} {'Accuracy':<10} {'VRAM(GB)':<10}"
    sep = "─" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for r in rows:
        def _fmt(v, fmt=".1f"):
            return f"{v:{fmt}}" if isinstance(v, (int, float)) else "N/A"

        logger.info(
            "%-22s %-10s %-12s %-10s %-10s %-10s",
            r["model"][:22],
            r["status"],
            _fmt(r.get("avg_ttft_ms")),
            _fmt(r.get("avg_throughput_tps")),
            _fmt(r.get("avg_accuracy"), ".4f"),
            _fmt(r.get("peak_vram_gb")),
        )
    logger.info(sep)


if __name__ == "__main__":
    main()
