#!/usr/bin/env python
"""Main entry point for multi-model, dual-runtime benchmarking.

This module orchestrates benchmark runs over a set of VLMs/LLMs, evaluating each
model on selected runtimes (HuggingFace, vLLM) with configurable metric suites.
Results are saved incrementally per (model, runtime) pair and aggregated into a
comparison summary. Metric routing respects runtime capabilities: perplexity
requires logit access (HF only), OCR evals require VLM inference, and other
metrics may run on both backends.

Metric support matrix:
  - quality_llm.ppl, logit_kl → HF only (logit access required)
  - quality_llm.mmlu_tiny, gsm8k, aime, ceval, humaneval → HF and vLLM
  - quality_ocr.* → HF only, VLM models only (generate_vlm method)
  - perf.*, memory.* → HF and vLLM

Usage:
  python benchmark.py --config config/benchmark/ocr_comparison.yaml
  python benchmark.py --config config/benchmark/ocr_comparison.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────
from src.utils import setup_global_logging
logger = setup_global_logging("benchmark")


from src.config import load_benchmark_config
from src.config.schemas import BenchmarkConfig, BenchmarkModelEntry
from src.reporting.extract import (
    bits_grid, ctx_rows, curated_scalars, flatten_metrics, tput_rows,
)

# Fixed prompt used for latency / throughput profiling.
_PERF_PROMPT = "Explain the theory of relativity in simple terms, step by step."
# Questions per MMLU subject (kept small so the eval stays fast).
_MMLU_Q_PER_SUBJECT = 50


# Helpers
def _safe_filename(name: str) -> str:
    """Convert display name to filesystem-safe filename stem.

    Lowercases and replaces spaces and slashes with underscores/dashes for
    use in output file and directory names.

    Args:
        name: Display name (may contain spaces, slashes, uppercase).

    Returns:
        Safe filename stem (lowercase, spaces → underscores, slashes → dashes).
    """
    return name.lower().replace(" ", "_").replace("/", "-")


def _dir_size_mb(path: Path) -> float:
    """Compute total disk size of a directory tree.

    Recursively sums file sizes under the directory, excluding directory
    entries themselves.

    Args:
        path: Directory path to measure.

    Returns:
        Total size in MB. For a local dir, sums its files. For a hub repo id
        (is_local=False), reads the model's size from the HF cache. 0.0 if neither.
    """
    if path.exists() and path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 ** 2), 2)
    # Not a local dir → treat as a hub repo id and read its cached size on disk.
    # Normalize backslashes: Path() rewrites "Qwen/Qwen3-1.7B" with the OS separator,
    # but repo_id always uses "/".
    repo_id = str(path).replace("\\", "/")
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                return round(repo.size_on_disk / (1024 ** 2), 2)
    except Exception:
        pass
    return 0.0


def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve benchmark model path to actual checkpoint directory.

    The quantize.py workflow saves models into subdirectories named
    {model}-{backend}-{method}-{scheme}. If the provided path is a parent
    directory lacking config.json but containing exactly one child with
    config.json, descend into it. Supports both direct checkpoints and
    their parent containers.

    Args:
        path: Checkpoint directory or parent container path.

    Returns:
        Resolved path to config.json directory (child if found, else original).
    """
    p = Path(path)
    if not p.is_dir() or (p / "config.json").exists():
        return path
    subdirs = [d for d in p.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if len(subdirs) == 1:
        logger.info("Resolved checkpoint dir: '%s' -> '%s'", path, subdirs[0])
        return str(subdirs[0])
    return path


# Metric groups (each is crash-safe at the caller level)
def _run_quality_llm(runtime, runtime_name: str, config: BenchmarkConfig) -> dict:
    """Compute text-quality metrics enabled for this benchmark run.

    Dispatches to per-metric evaluators (perplexity, MMLU, GSM8K, AIME, C-Eval,
    HumanEval) based on config.metrics.quality_llm. Skips logit-dependent metrics
    on vLLM (which does not expose raw logits). Returns early-exit markers for
    unimplemented metrics (logit_kl, token_agree).

    Args:
        runtime: Initialized model runtime (HF or vLLM).
        runtime_name: "hf" or "vllm" for logging and conditional metric gating.
        config: BenchmarkConfig controlling dataset params, seeds, sample counts.

    Returns:
        Dictionary mapping metric name (str) to result dict or skip marker.
        Keys: "ppl", "mmlu_acc", "gsm8k", "aime", "ceval", "humaneval", etc.
    """
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

    # ── Reasoning / code / Chinese-MC evals (eval_tasks) ──────────────────────
    # Generative (gsm8k/aime/humaneval) run on ANY runtime via generate(); C-Eval
    # uses score_choices() which both HF and vLLM implement. Only ppl (above) is
    # genuinely HF-only (needs raw logits).
    ds = config.datasets
    new_evals = {"gsm8k", "ceval", "humaneval", "aime"} & set(wanted)
    if new_evals:
        from src.evaluation.eval_tasks import (
            eval_gsm8k, eval_ceval, eval_humaneval, eval_aime,
        )
        if "gsm8k" in wanted:
            out["gsm8k"] = eval_gsm8k(
                runtime, num_samples=ds.gsm8k_num_samples,
                max_new_tokens=ds.gsm8k_max_new_tokens, seed=config.seed,
                dataset_name=ds.gsm8k_dataset,
            )
        if "aime" in wanted:
            out["aime"] = eval_aime(
                runtime, num_samples=ds.aime_num_samples,
                max_new_tokens=ds.aime_max_new_tokens, seed=config.seed,
                dataset_name=ds.aime_dataset, split=ds.aime_split,
            )
        if "ceval" in wanted:
            out["ceval"] = eval_ceval(
                runtime, subjects=ds.ceval_subjects,
                num_q_per_subject=ds.ceval_num_q_per_subject,
                seed=config.seed, dataset_name=ds.ceval_dataset,
            )
        if "humaneval" in wanted:
            out["humaneval"] = eval_humaneval(
                runtime, num_samples=ds.humaneval_num_samples,
                max_new_tokens=ds.humaneval_max_new_tokens, seed=config.seed,
                execute=ds.humaneval_execute, timeout=ds.humaneval_timeout,
                dataset_name=ds.humaneval_dataset,
            )

    for m in ("logit_kl", "token_agree"):
        if m in wanted:
            out[m] = {"skipped": "requires baseline wiring (not yet implemented)"}

    return out


def _run_quality_ocr(runtime, runtime_name: str, entry: BenchmarkModelEntry,
                     config: BenchmarkConfig) -> dict:
    """Evaluate OCR performance on vision-language models.

    Only executes if the model is a VLM and the runtime supports generate_vlm().
    Evaluates one or more OCR datasets (specified in config.datasets.ocr_dataset
    or .ocr_datasets) and returns results keyed by dataset name if multiple.

    Args:
        runtime: Initialized VLM runtime.
        runtime_name: Runtime identifier for conditional gating (HF only).
        entry: BenchmarkModelEntry with is_vlm flag and model metadata.
        config: BenchmarkConfig with OCR dataset and metric parameters.

    Returns:
        Single dataset → flat dict with keys (cer, wer, exact_match, bleu, num_samples).
        Multiple datasets → dict[dataset_name] of same structure.
        Returns skip marker dict if model is not VLM or runtime unsupported.
    """
    if not entry.is_vlm:
        return {"skipped": "model_type != 'vlm'"}
    if runtime_name != "hf":
        return {"skipped": "OCR (generate_vlm) supported on HF runtime only"}

    from src.evaluation.eval_ocr import eval_ocr

    ds = config.datasets
    names = ds.ocr_datasets or [ds.ocr_dataset]

    def _one(name: str) -> dict:
        r = eval_ocr(
            runtime,
            dataset_name=name,
            num_samples=ds.ocr_num_samples,
            max_new_tokens=ds.ocr_max_new_tokens,
            split=ds.ocr_split,
            seed=config.seed,
            image_field=ds.ocr_image_field,
            text_field=ds.ocr_text_field,
            subset=ds.ocr_subset,
        )
        r.pop("per_sample", None)   # keep aggregate metrics only
        return r

    # Single dataset → flat dict (keeps comparison-summary cer extraction working).
    if len(names) == 1:
        return _one(names[0])
    # Multiple datasets → keyed by dataset name.
    return {name: _one(name) for name in names}


def _run_perf(runtime, entry: BenchmarkModelEntry, config: BenchmarkConfig) -> dict:
    """Benchmark performance metrics (latency, throughput, context capacity).

    Delegates to the OOM-isolated performance runner. The TQ bits/accuracy sweep
    only runs for TurboQuant-enabled entries (it characterizes TQ, identical on the
    baseline), so it isn't duplicated.
    """
    from src.evaluation.performance import run_perf_metrics
    return run_perf_metrics(runtime, config, perf_prompt=_PERF_PROMPT,
                            tq_enabled=entry.turboquant_enabled)


def _run_memory(runtime, entry: BenchmarkModelEntry,
                load_time_s: float, config: BenchmarkConfig) -> dict:
    """Collect memory and disk footprint metrics.

    Args:
        runtime: Initialized model runtime with peak_vram_mb() method.
        entry: BenchmarkModelEntry pointing to the model checkpoint.
        load_time_s: Model load duration in seconds.
        config: BenchmarkConfig with memory.* metric selection flags.

    Returns:
        Dict with keys: disk_mb (checkpoint size), peak_vram_mb (max GPU memory),
        load_time_s (model initialization duration). Only present keys correspond
        to enabled metrics in config.metrics.memory.
    """
    out: dict = {}
    wanted = config.metrics.memory

    if "disk" in wanted:
        out["disk_mb"] = _dir_size_mb(Path(entry.path))
    if "vram" in wanted:
        out["peak_vram_mb"] = round(runtime.peak_vram_mb(), 2)
        if hasattr(runtime, "model_vram_mb"):
            out["model_vram_mb"] = round(runtime.model_vram_mb(), 2)   # resident weights
    if "load_time" in wanted:
        out["load_time_s"] = round(load_time_s, 2)

    return out


def _run_model_on_runtime(
    entry: BenchmarkModelEntry,
    runtime_name: str,
    config: BenchmarkConfig,
) -> dict:
    """Load a model and run all enabled metric suites, with per-group error isolation.

    Instantiates the runtime, loads the model, runs each metric group
    (memory, quality_llm, quality_ocr, perf) in a try-except wrapper so one
    group's failure does not prevent others from running. Model is always
    unloaded in a finally block. GPU OOM during load is caught and markers
    are returned for graceful degradation.

    Args:
        entry: BenchmarkModelEntry with model metadata and path.
        runtime_name: "hf" or "vllm" runtime identifier.
        config: BenchmarkConfig controlling all evaluation parameters.

    Returns:
        Dict with keys: "memory", "quality_llm", "quality_ocr", "perf" containing
        metric results or error/skip markers. Special markers: "oom_on_load",
        "error_on_load", "fatal_error" indicate load-stage failures.

    Raises:
        Does not raise; exceptions are caught and recorded in output dict.
    """
    from src.runtimes import build_runtime
    from src.evaluation.performance import clear_gpu_memory, is_oom_error

    results: dict = {}

    logger.info("[%s @ %s] Loading model ...", entry.name, runtime_name)
    t0 = time.perf_counter()
    try:
        runtime = build_runtime(runtime_name, entry)   # instantiates + loads
    except Exception as exc:
        # Load-time OOM/error must not poison the next model — free VRAM and
        # return a marker so the outer loop records "failed" and moves on.
        clear_gpu_memory()
        marker = "oom_on_load" if is_oom_error(exc) else "error_on_load"
        logger.error("[%s @ %s] %s: %s", entry.name, runtime_name, marker, exc)
        return {marker: traceback.format_exc()}
    load_time_s = time.perf_counter() - t0

    try:
        groups = {
            "memory":      lambda: _run_memory(runtime, entry, load_time_s, config),
            "quality_llm": lambda: _run_quality_llm(runtime, runtime_name, config),
            "quality_ocr": lambda: _run_quality_ocr(runtime, runtime_name, entry, config),
            "perf":        lambda: _run_perf(runtime, entry, config),
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


# ── Weights & Biases tracking ────────────────────────────────────────────────
def _init_wandb(config: BenchmarkConfig):
    """Build a WandBLogger if W&B tracking is enabled and importable, else None.

    Single project (config.tracking.wandb_project); each (model, runtime) becomes
    its own run. Never raises — tracking failures must not abort the benchmark.
    """
    import os
    if "wandb" not in config.tracking.enabled:
        return None
    # Only activate when credentials exist, so a no-creds run never blocks on an
    # interactive wandb login prompt. Honour WANDB_API_KEY (or an existing login).
    has_key = bool(os.environ.get(config.tracking.wandb_api_key_env))
    if not has_key:
        try:
            import wandb
            has_key = wandb.api.api_key is not None
        except Exception:
            has_key = False
    if not has_key:
        logger.warning("W&B enabled but no API key (%s) and not logged in — "
                       "skipping tracking. `wandb login` or set the env var to enable.",
                       config.tracking.wandb_api_key_env)
        return None
    try:
        from src.tracking.wandb_connector import WandBLogger
        # track_gpu=False: the per-run GPU background thread logs with auto-steps,
        # which races with our explicit-step scalar logs and causes wandb to drop
        # them ("not monotonic" → empty charts). Peak VRAM is already captured in
        # the memory metrics, so the thread adds little here.
        return WandBLogger(
            project_name=config.tracking.wandb_project,
            entity=config.tracking.wandb_entity,
            track_gpu=False,
        )
    except Exception as exc:
        logger.warning("W&B tracking disabled (init failed): %s", exc)
        return None


def _log_wandb_charts(run, metrics: dict) -> None:
    """Log the curated W&B charts: context sweep, TQ bits sweep, throughput."""
    import wandb
    panels: dict = {}

    rows = ctx_rows(metrics)
    if rows:
        lens = [r["context_len"] for r in rows]
        panels["ctx_sweep/kv_cache_mb_vs_len"] = wandb.plot.line_series(
            xs=lens, keys=["turboquant", "fp16"],
            ys=[[r["kv_cache_mb"] for r in rows], [r.get("fp16_kv_mb") for r in rows]],
            title="Resident KV cache (MB) vs context length", xname="context_len")
        ctx_table = wandb.Table(
            columns=["context_len", "kv_cache_mb", "fp16_kv_mb", "compression_ratio", "peak_vram_mb"],
            data=[[r["context_len"], r["kv_cache_mb"], r.get("fp16_kv_mb"),
                   r.get("compression_ratio"), r.get("peak_vram_mb")] for r in rows])
        panels["ctx_sweep/compression_vs_len"] = wandb.plot.line(
            ctx_table, "context_len", "compression_ratio", title="KV compression ratio vs context length")
        panels["ctx_sweep/peak_vram_vs_len"] = wandb.plot.line(
            ctx_table, "context_len", "peak_vram_mb", title="Peak VRAM (MB) vs context length")
        panels["ctx_sweep/table"] = ctx_table

    grid = bits_grid(metrics)
    if grid:
        lengths = sorted({g["context_len"] for g in grid})
        bits = sorted({g["bits"] for g in grid})

        def _series(field):
            by = {b: {} for b in bits}
            for g in grid:
                by[g["bits"]][g["context_len"]] = g.get(field)
            return [[by[b].get(L) for L in lengths] for b in bits]

        panels["tq_bits/top5_agreement_vs_len"] = wandb.plot.line_series(
            xs=lengths, ys=_series("top5_agreement"), keys=bits,
            title="Top-5 next-token agreement vs FP16 (by bits)", xname="context_len")
        panels["tq_bits/top1_agreement_vs_len"] = wandb.plot.line_series(
            xs=lengths, ys=_series("top1_agreement"), keys=bits,
            title="Top-1 next-token agreement vs FP16 (by bits)", xname="context_len")
        panels["tq_bits/kl_div_vs_len"] = wandb.plot.line_series(
            xs=lengths, ys=_series("kl_div"), keys=bits,
            title="KL(FP16 || TQ) vs context length (by bits)", xname="context_len")
        panels["tq_bits/kv_cache_mb_vs_len"] = wandb.plot.line_series(
            xs=lengths, ys=_series("kv_cache_mb"), keys=bits,
            title="Resident KV cache (MB) vs context length (by bits)", xname="context_len")
        panels["tq_bits/table"] = wandb.Table(
            columns=["bits", "context_len", "top1_agreement", "top5_agreement",
                     "kl_div", "kv_cache_mb", "compression_ratio"],
            data=[[g["bits"], g["context_len"], g["top1_agreement"], g.get("top5_agreement"),
                   g.get("kl_div"), g.get("kv_cache_mb"), g.get("compression_ratio")] for g in grid])

    tput = tput_rows(metrics)
    if tput:
        tp_table = wandb.Table(
            columns=["batch_size", "tokens_per_sec", "latency_ms", "turboquant"],
            data=[[r["batch_size"], r.get("tokens_per_sec"), r.get("latency_ms"),
                   bool(r.get("turboquant"))] for r in tput])
        panels["throughput/tps_vs_batch"] = wandb.plot.line(
            tp_table, "batch_size", "tokens_per_sec", title="Throughput (tok/s) vs batch size")
        panels["throughput/table"] = tp_table

    if panels:
        run.log(panels)


def _log_record_to_wandb(tracker, record: dict, entry: BenchmarkModelEntry,
                         runtime_name: str, config: BenchmarkConfig, arch: str) -> None:
    """Log one (model, runtime) record as a dedicated W&B run. Best-effort.

    Scalars go to run.summary (the cross-model comparison table) only — NOT
    run.log — so each scalar doesn't spawn its own single-point time-series panel.
    Curated sweep charts are the only logged panels.
    """
    if tracker is None:
        return
    run_name = f"{entry.name} @ {runtime_name}" + (" +tq" if entry.turboquant_enabled else "")
    try:
        run_cfg = {
            "model_name": entry.name, "model_path": entry.path,
            "model_type": entry.model_type, "model_class": entry.model_class,
            "runtime": runtime_name, "dtype": entry.dtype,
            "hf_quantization": entry.hf_quantization,
            "vllm_quantization": entry.vllm_quantization,
            "turboquant": entry.turboquant.model_dump() if entry.turboquant else None,
            "gpu_arch": arch, "benchmark_run": config.run_name,
        }
        tracker.start_run(run_name=run_name, config=run_cfg)
        run = tracker._current_run
        if run is not None:
            m = record.get("metrics", {})
            # Curated headline scalars as comparison panels; full set in summary.
            run.log(curated_scalars(m))
            run.summary.update(flatten_metrics(m))
            run.summary["status"] = record.get("status")
            _log_wandb_charts(run, m)
        tracker.finish_current_run()
    except Exception as exc:
        logger.warning("W&B logging failed for '%s': %s", run_name, exc)


def _log_record_to_tensorboard(tb_dir: Path, tag: str, record: dict) -> None:
    """Mirror a record to TensorBoard: flat scalars + sweep curves (step = x-axis)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return
    import re
    safe = re.sub(r"[^0-9A-Za-z._@+-]+", "_", tag)
    try:
        w = SummaryWriter(log_dir=str(tb_dir / safe))
        m = record.get("metrics", {})
        for k, v in flatten_metrics(m).items():
            w.add_scalar(k, v)
        for s in ctx_rows(m):
            L = s["context_len"]
            w.add_scalar("ctx_sweep/kv_cache_mb", s["kv_cache_mb"], L)
            w.add_scalar("ctx_sweep/peak_vram_mb", s["peak_vram_mb"], L)
            w.add_scalar("ctx_sweep/compression_ratio", s.get("compression_ratio", 1.0), L)
        for g in bits_grid(m):
            L, b = g["context_len"], g["bits"]
            w.add_scalar(f"tq_bits_top1/{b}", g["top1_agreement"], L)
            w.add_scalar(f"tq_bits_top5/{b}", g.get("top5_agreement", 0.0), L)
            w.add_scalar(f"tq_bits_kv_mb/{b}", g.get("kv_cache_mb", 0.0), L)
        for r in tput_rows(m):
            w.add_scalar("throughput/tokens_per_sec", r["tokens_per_sec"], r["batch_size"])
        w.close()
    except Exception as exc:
        logger.warning("TensorBoard logging failed for '%s': %s", tag, exc)


# Summary
def _build_comparison_summary(all_results: dict, environment: dict | None = None,
                              run_name: str | None = None) -> dict:
    """Reshape flat benchmark results into a comparison table.

    Flattens per-(model, runtime) result records, extracting key metrics
    (latency, throughput, VRAM, PPL, MMLU, CER) into columnar rows for
    easy comparison across models and runtimes.

    Args:
        all_results: Dict mapping result keys (e.g., "modelA@hf") to full
                     result records from _run_model_on_runtime().
        environment: Optional hardware/software snapshot from _capture_environment().
        run_name:    Optional BenchmarkConfig.run_name, for provenance.

    Returns:
        Dict with keys:
          - timestamp: ISO datetime of benchmark run start
          - run_name: the benchmark config's run_name (or None)
          - environment: GPU/CUDA/driver/torch/transformers/commit snapshot (or None)
          - num_records: count of result rows
          - records: list of dicts (one per model-runtime pair) with columns
                     [model, runtime, status, ttft_ms_p50, tpot_ms_p50, best_tps,
                      peak_vram_mb, disk_mb, ppl, mmlu_acc, ocr_cer]
    """
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
        "run_name": run_name,
        "environment": environment,
        "num_records": len(rows),
        "records": rows,
    }


def _print_summary_table(summary: dict) -> None:
    """Pretty-print the comparison table to the log.

    Metric columns are adaptive: an optional column (PPL/MMLU/CER/…) is shown
    only when at least one row carries a value for it. So an OCR-only run drops
    PPL/MMLU, and an LLM-only run drops CER — no all-N/A columns.
    """
    rows = summary.get("records", [])
    if not rows:
        return

    def _fmt(v, fmt=".1f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else "N/A"

    # (key, header, value-fmt, column-width). First three are always shown.
    core_cols = [
        ("model",   "Model",  None, 20),
        ("runtime", "RT",     None, 6),
        ("status",  "Status", None, 9),
    ]
    optional_cols = [
        ("ttft_ms_p50",  "TTFT",     ".1f", 9),
        ("tpot_ms_p50",  "TPOT",     ".1f", 9),
        ("best_tps",     "TPS",      ".1f", 10),
        ("peak_vram_mb", "VRAM(MB)", ".1f", 11),
        ("ppl",          "PPL",      ".2f", 8),
        ("mmlu_acc",     "MMLU",     ".3f", 7),
        ("ocr_cer",      "CER",      ".3f", 7),
    ]
    # Keep only optional columns that have data in at least one row.
    shown = [c for c in optional_cols
             if any(isinstance(r.get(c[0]), (int, float)) for r in rows)]
    cols = core_cols + shown

    header = "".join(f"{h:<{w}}" for _, h, _, w in cols)
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for r in rows:
        line = ""
        for key, _h, fmt, w in cols:
            if fmt is None:
                cell = str(r.get(key))
                if key == "model":
                    cell = cell[:w - 1]
            else:
                cell = _fmt(r.get(key), fmt)
            line += f"{cell:<{w}}"
        logger.info(line)
    logger.info(sep)



# Main
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

    logger.info("-" * 60)
    logger.info("Benchmark plan: %s", config.run_name)
    for i, m in enumerate(config.models, 1):
        logger.info("  [%d] %-20s -> %s (%s)", i, m.name, m.path, m.model_type)
    logger.info("  Runtimes : %s", config.runtimes)
    logger.info("  Metrics  : llm=%s ocr=%s perf=%s mem=%s",
                config.metrics.quality_llm, config.metrics.quality_ocr,
                config.metrics.perf, config.metrics.memory)
    logger.info("  Results  : %s/", results_dir)
    logger.info("-" * 60)

    if args.dry_run:
        logger.info("--dry-run: config valid [OK] - stopping before model load.")
        return

    all_results: dict[str, dict] = {}
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arch = _detect_arch()
    environment = _capture_environment()
    logger.info("  Environment: gpu=%s driver=%s cuda=%s torch=%s commit=%s",
                environment.get("gpu_name"), environment.get("gpu_driver_version"),
                environment.get("cuda_version"), environment.get("torch_version"),
                environment.get("git_commit"))
    tracker = _init_wandb(config)
    if tracker is not None:
        logger.info("W&B tracking on -> project '%s'", config.tracking.wandb_project)
    tb_dir = results_dir / "tensorboard" if "tensorboard" in config.tracking.enabled else None
    if tb_dir is not None:
        logger.info("TensorBoard logging on -> %s  (view: tensorboard --logdir %s)",
                    tb_dir, tb_dir)

    for entry in config.models:
        if arch and arch in entry.skip_on:
            logger.warning("Skipping '%s' on arch=%s (skip_on).", entry.name, arch)
            continue

        # Pre-skip local checkpoints that haven't been produced yet — avoids a noisy
        # from_pretrained traceback per missing model. Hub ids (is_local=False) fall
        # through and download as usual.
        if entry.is_local and not Path(entry.path).exists():
            logger.warning("Skipping '%s': local path not found (%s). "
                           "Quantize this variant first.", entry.name, entry.path)
            all_results[f"{entry.name}@_"] = {
                "model_name": entry.name, "model_path": entry.path,
                "model_type": entry.model_type, "runtime": "—",
                "status": "skipped", "timestamp": run_ts,
                "environment": environment,
                "metrics": {"skipped": "local checkpoint not found"},
            }
            continue

        # Descend into the quantize.py save subfolder if the path points at its parent.
        if entry.is_local:
            entry.path = _resolve_checkpoint_dir(entry.path)

        for runtime_name in config.runtimes:
            tag = f"{entry.name}@{runtime_name}"
            logger.info("=" * 60)
            logger.info("▶  %s", tag)
            logger.info("=" * 60)

            try:
                metrics = _run_model_on_runtime(entry, runtime_name, config)
                # _run_model_on_runtime returns a marker dict (not an exception)
                # when the model never loaded — reflect that in the status.
                if any(k in metrics for k in ("oom_on_load", "error_on_load", "fatal_error")):
                    status = "failed"
                else:
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
                "environment": environment,
                "metrics":    metrics,
            }
            all_results[tag] = record

            out_path = results_dir / f"{_safe_filename(entry.name)}_{runtime_name}_{run_ts}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
            logger.info("Saved -> %s", out_path)

            _log_record_to_wandb(tracker, record, entry, runtime_name, config, arch)
            if tb_dir is not None:
                _log_record_to_tensorboard(tb_dir, f"{entry.name}@{runtime_name}", record)

    summary = _build_comparison_summary(all_results, environment, config.run_name)
    summary_path = results_dir / f"comparison_summary_{run_ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Benchmark complete. Summary -> %s", summary_path)
    logger.info("=" * 60)
    _print_summary_table(summary)

    if tracker is not None:
        try:
            tracker.__exit__(None, None, None)   # finish run + NVML shutdown
        except Exception:
            pass


def _detect_arch() -> str:
    """Return GPU arch string (e.g. 'sm_86') for skip_on filtering; '' on failure."""
    try:
        from src.utils import get_gpu_arch
        return get_gpu_arch()
    except Exception:
        return ""


def _git_commit() -> str | None:
    """Short git commit hash of the checkout running this benchmark, or None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _nvidia_driver_version() -> str | None:
    """Driver version via nvidia-smi, or None (non-NVIDIA / not installed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _capture_environment() -> dict:
    """Snapshot the hardware/software environment once per benchmark run.

    Reproducibility record (GPU, CUDA, driver, torch/transformers, commit) so a
    results JSON is self-describing without cross-referencing when/where it ran.
    Every field is best-effort; a failed lookup is None rather than aborting the run.
    """
    env: dict = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "git_commit": _git_commit(),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_total_vram_mb"] = round(props.total_memory / (1024 ** 2), 1)
            env["gpu_driver_version"] = _nvidia_driver_version()
        else:
            env["gpu_name"] = None
    except Exception as exc:
        logger.warning("Environment capture: torch/cuda info incomplete: %s", exc)
    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except Exception:
        env["transformers_version"] = None
    return env


if __name__ == "__main__":
    main()
