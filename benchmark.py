#!/usr/bin/env python
"""
benchmark.py — TripleQuant-VLM multi-model, dual-runtime benchmark entry point.

For every model in the config, runs the enabled metric groups on each selected
runtime (HF and/or vLLM), saves per-(model, runtime) JSON immediately
(crash-safe), then writes a final comparison summary.

Metric routing (a metric is skipped — with a logged note — when the runtime
cannot support it):
    quality_llm.ppl / logit_kl   → HF only (need raw logits; vLLM hides them)
    quality_llm.mmlu_tiny        → HF + vLLM (log-prob scoring)
    quality_ocr.*                → HF only, VLM models only (generate_vlm)
    perf.*                       → HF + vLLM
    memory.*                     → HF + vLLM

Usage:
    python benchmark.py --config config/benchmark/ocr_comparison.yaml
    python benchmark.py --config config/benchmark/ocr_comparison.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from src.config.schemas import BenchmarkConfig, BenchmarkModelEntry

# Fixed prompt used for latency / throughput profiling.
_PERF_PROMPT = "Explain the theory of relativity in simple terms, step by step."
# Questions per MMLU subject (kept small so the eval stays fast).
_MMLU_Q_PER_SUBJECT = 50


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Convert a display name into a safe filename stem."""
    return name.lower().replace(" ", "_").replace("/", "-")


def _dir_size_mb(path: Path) -> float:
    """Total size (MB) of all files under a directory, 0.0 if not a local dir."""
    if not path.exists() or not path.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 ** 2), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Metric groups (each is crash-safe at the caller level)
# ─────────────────────────────────────────────────────────────────────────────

def _run_quality_llm(runtime, runtime_name: str, config: BenchmarkConfig) -> dict:
    """Run enabled LLM quality metrics (ppl, mmlu_tiny). Returns partial dict."""
    from src.evaluation.eval_llm import compute_ppl, eval_mmlu_tiny
    from src.utils import load_wikitext

    out: dict = {}
    wanted = config.metrics.quality_llm

    if "ppl" in wanted:
        if runtime_name == "hf":
            ds = load_wikitext(
                version=config.datasets.ppl_subset,
                split="test",
                num_samples=256,
                seed=config.seed,
            )
            out["ppl"] = compute_ppl(runtime, ds, max_chunks=64)
        else:
            out["ppl"] = {"skipped": "ppl needs logits; vLLM unsupported"}

    if "mmlu_tiny" in wanted:
        out["mmlu_acc"] = eval_mmlu_tiny(
            runtime,
            config.datasets.mmlu_subjects,
            num_q_per_subject=_MMLU_Q_PER_SUBJECT,
            seed=config.seed,
        )

    for m in ("logit_kl", "token_agree"):
        if m in wanted:
            out[m] = {"skipped": "requires baseline wiring (not yet implemented)"}

    return out


def _run_quality_ocr(runtime, runtime_name: str, entry: BenchmarkModelEntry,
                     config: BenchmarkConfig) -> dict:
    """Run OCR metrics for VLM models on a generate_vlm-capable runtime."""
    if not entry.is_vlm:
        return {"skipped": "model_type != 'vlm'"}
    if runtime_name != "hf":
        return {"skipped": "OCR (generate_vlm) supported on HF runtime only"}

    from src.evaluation.eval_ocr import eval_ocr

    results = eval_ocr(
        runtime,
        dataset_name=config.datasets.ocr_dataset,
        num_samples=config.datasets.ocr_num_samples,
        max_new_tokens=config.datasets.ocr_max_new_tokens,
        seed=config.seed,
    )
    # Drop heavy per-sample list from the saved summary (kept metrics only).
    results.pop("per_sample", None)
    return results


def _run_perf(runtime, config: BenchmarkConfig) -> dict:
    """Run latency (ttft/tpot) and throughput sweeps."""
    out: dict = {}
    wanted = config.metrics.perf
    lat = config.latency

    if "ttft" in wanted or "tpot" in wanted:
        out["latency"] = runtime.measure_ttft_tpot(
            _PERF_PROMPT, n=lat.num_requests,
        )

    if "throughput" in wanted:
        out["throughput"] = runtime.measure_throughput(
            _PERF_PROMPT,
            batch_sizes=lat.batch_sizes,
            output_len=lat.output_lens[0],
        )

    return out


def _run_memory(runtime, entry: BenchmarkModelEntry,
                load_time_s: float, config: BenchmarkConfig) -> dict:
    """Collect disk size, peak VRAM, and load time."""
    out: dict = {}
    wanted = config.metrics.memory

    if "disk" in wanted:
        out["disk_mb"] = _dir_size_mb(Path(entry.path))
    if "vram" in wanted:
        out["peak_vram_mb"] = round(runtime.peak_vram_mb(), 2)
    if "load_time" in wanted:
        out["load_time_s"] = round(load_time_s, 2)

    return out


def _run_model_on_runtime(
    entry: BenchmarkModelEntry,
    runtime_name: str,
    config: BenchmarkConfig,
) -> dict:
    """Build the runtime for *entry*, run all enabled metric groups, unload.

    Each metric group is wrapped so one failure does not abort the others;
    failed groups map to ``{"error": "<traceback>"}``.
    """
    from src.runtimes import build_runtime

    results: dict = {}

    logger.info("[%s @ %s] Loading model …", entry.name, runtime_name)
    t0 = time.perf_counter()
    runtime = build_runtime(runtime_name, entry)   # instantiates + loads
    load_time_s = time.perf_counter() - t0

    try:
        groups = {
            "memory":      lambda: _run_memory(runtime, entry, load_time_s, config),
            "quality_llm": lambda: _run_quality_llm(runtime, runtime_name, config),
            "quality_ocr": lambda: _run_quality_ocr(runtime, runtime_name, entry, config),
            "perf":        lambda: _run_perf(runtime, config),
        }
        for group_name, fn in groups.items():
            logger.info("[%s @ %s] metric group: %s", entry.name, runtime_name, group_name)
            try:
                results[group_name] = fn()
            except Exception:
                logger.warning("[%s @ %s] group '%s' failed.",
                               entry.name, runtime_name, group_name, exc_info=True)
                results[group_name] = {"error": traceback.format_exc()}
    finally:
        runtime.unload()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _build_comparison_summary(all_results: dict) -> dict:
    """Flatten per-(model,runtime) records into side-by-side rows."""
    rows = []
    for key, record in all_results.items():
        m = record.get("metrics", {})
        perf = m.get("perf", {})
        lat = perf.get("latency", {}) if isinstance(perf, dict) else {}
        tput = perf.get("throughput", []) if isinstance(perf, dict) else []
        mem = m.get("memory", {}) if isinstance(m.get("memory"), dict) else {}
        qllm = m.get("quality_llm", {}) if isinstance(m.get("quality_llm"), dict) else {}
        qocr = m.get("quality_ocr", {}) if isinstance(m.get("quality_ocr"), dict) else {}

        best_tps = None
        if isinstance(tput, list) and tput:
            tps_vals = [r.get("tokens_per_sec", 0.0) for r in tput if not r.get("oom")]
            best_tps = max(tps_vals) if tps_vals else None

        rows.append({
            "model":          record.get("model_name"),
            "runtime":        record.get("runtime"),
            "status":         record.get("status"),
            "ttft_ms_p50":    lat.get("ttft_ms_p50"),
            "tpot_ms_p50":    lat.get("tpot_ms_p50"),
            "best_tps":       best_tps,
            "peak_vram_mb":   mem.get("peak_vram_mb"),
            "disk_mb":        mem.get("disk_mb"),
            "ppl":            qllm.get("ppl") if isinstance(qllm.get("ppl"), (int, float)) else None,
            "mmlu_acc":       qllm.get("mmlu_acc"),
            "ocr_cer":        qocr.get("cer"),
        })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_records": len(rows),
        "records": rows,
    }


def _print_summary_table(summary: dict) -> None:
    """Pretty-print the comparison table to the log."""
    rows = summary.get("records", [])
    if not rows:
        return

    def _fmt(v, fmt=".1f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else "N/A"

    header = (f"{'Model':<20}{'RT':<6}{'Status':<9}{'TTFT':<9}{'TPOT':<9}"
              f"{'TPS':<10}{'VRAM(MB)':<11}{'PPL':<8}{'MMLU':<7}{'CER':<7}")
    sep = "─" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for r in rows:
        logger.info(
            f"{str(r['model'])[:19]:<20}{str(r['runtime']):<6}{str(r['status']):<9}"
            f"{_fmt(r.get('ttft_ms_p50')):<9}{_fmt(r.get('tpot_ms_p50')):<9}"
            f"{_fmt(r.get('best_tps')):<10}{_fmt(r.get('peak_vram_mb')):<11}"
            f"{_fmt(r.get('ppl'), '.2f'):<8}{_fmt(r.get('mmlu_acc'), '.3f'):<7}"
            f"{_fmt(r.get('ocr_cer'), '.3f'):<7}"
        )
    logger.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TripleQuant-VLM — benchmark multiple models on HF and/or vLLM.",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to a benchmark YAML config (e.g. config/benchmark/ocr_comparison.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print the plan without loading any model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading benchmark config: %s", args.config)
    try:
        config = load_benchmark_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    results_dir = Path(config.output_root) / config.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("─" * 60)
    logger.info("Benchmark plan: %s", config.run_name)
    for i, m in enumerate(config.models, 1):
        logger.info("  [%d] %-20s → %s (%s)", i, m.name, m.path, m.model_type)
    logger.info("  Runtimes : %s", config.runtimes)
    logger.info("  Metrics  : llm=%s ocr=%s perf=%s mem=%s",
                config.metrics.quality_llm, config.metrics.quality_ocr,
                config.metrics.perf, config.metrics.memory)
    logger.info("  Results  : %s/", results_dir)
    logger.info("─" * 60)

    if args.dry_run:
        logger.info("--dry-run: config valid ✓ — stopping before model load.")
        return

    all_results: dict[str, dict] = {}
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arch = _detect_arch()

    for entry in config.models:
        if arch and arch in entry.skip_on:
            logger.warning("Skipping '%s' on arch=%s (skip_on).", entry.name, arch)
            continue

        for runtime_name in config.runtimes:
            tag = f"{entry.name}@{runtime_name}"
            logger.info("=" * 60)
            logger.info("▶  %s", tag)
            logger.info("=" * 60)

            try:
                metrics = _run_model_on_runtime(entry, runtime_name, config)
                status = "success"
            except Exception:
                tb = traceback.format_exc()
                logger.error("[%s] Fatal:\n%s", tag, tb)
                metrics = {"fatal_error": tb}
                status = "failed"

            record = {
                "model_name": entry.name,
                "model_path": entry.path,
                "model_type": entry.model_type,
                "runtime":    runtime_name,
                "status":     status,
                "timestamp":  run_ts,
                "metrics":    metrics,
            }
            all_results[tag] = record

            out_path = results_dir / f"{_safe_filename(entry.name)}_{runtime_name}_{run_ts}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
            logger.info("💾 Saved → %s", out_path)

    summary = _build_comparison_summary(all_results)
    summary_path = results_dir / f"comparison_summary_{run_ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("✅ Benchmark complete. Summary → %s", summary_path)
    logger.info("=" * 60)
    _print_summary_table(summary)


def _detect_arch() -> str:
    """Return GPU arch string (e.g. 'sm_86') for skip_on filtering; '' on failure."""
    try:
        from src.utils import get_gpu_arch
        return get_gpu_arch()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
