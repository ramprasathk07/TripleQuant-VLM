#!/usr/bin/env python
"""Publish one consolidated W&B leaderboard run from saved benchmark results.

The benchmark itself logs one W&B run per (model, runtime) — good provenance,
bad comparison UX. This script reads a results directory (latest
comparison_summary_*.json + optional vllm_serving.jsonl) and logs a SINGLE run
holding:

  - leaderboard table (every model x every metric, HF + vLLM serving columns)
  - one bar chart per headline metric across models (decode TPS HF vs vLLM,
    peak VRAM, PPL, MMLU)

Usage (from the quantization env, WANDB_API_KEY via .env):
    python scripts/wandb_leaderboard.py --dir results/qwen3-1.7b-leaderboard
    python scripts/wandb_leaderboard.py --dir results/qwen3-1.7b-leaderboard --project triplequant-bench
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()


def _load_serving(results_dir: Path) -> dict[str, dict]:
    """vllm_serving.jsonl -> {row-name: result}, empty if the file isn't there."""
    f = results_dir / "vllm_serving.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            out[rec.get("model", "?")] = rec
        except json.JSONDecodeError:
            continue
    return out


def _serving_for(model_name: str, serving: dict[str, dict]) -> dict | None:
    """Match a benchmark model name to its vllm_serving row (short-name keys)."""
    for key, rec in serving.items():
        if key in model_name or model_name.endswith(key):
            return rec
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Consolidated W&B leaderboard run")
    p.add_argument("--dir", "-d", required=True, help="results/<run_name> directory")
    p.add_argument("--project", default="triplequant-bench")
    p.add_argument("--run-name", default=None, help="W&B run name (default: <run_name>-leaderboard)")
    args = p.parse_args()

    results_dir = Path(args.dir)
    from src.reporting.loader import load_report_data
    data = load_report_data(results_dir)
    serving = _load_serving(results_dir)

    import wandb

    run = wandb.init(
        project=args.project,
        name=args.run_name or f"{data.run_name or results_dir.name}-leaderboard",
        job_type="leaderboard",
        config={"source": str(data.summary_path), "environment": data.environment},
    )

    cols = ["model", "status", "decode_tps_hf", "decode_tps_vllm", "ttft_ms_hf",
            "ttft_ms_vllm", "peak_vram_mb", "disk_mb", "ppl", "mmlu_acc", "serving"]
    table = wandb.Table(columns=cols)
    num = (int, float)

    for r in data.summary_rows:
        s = _serving_for(r.get("model", ""), serving)
        if s is None:
            serving_str = "not benched"
        elif s.get("loaded"):
            serving_str = "vLLM OK"
        else:
            serving_str = f"vLLM refused: {(s.get('error') or '')[:80]}"
        table.add_data(
            r.get("model"), r.get("status"),
            r.get("best_tps"), s.get("decode_tps") if s else None,
            r.get("ttft_ms_p50"), s.get("ttft_ms") if s else None,
            r.get("peak_vram_mb"), r.get("disk_mb"),
            r.get("ppl"), r.get("mmlu_acc"), serving_str,
        )

    panels: dict = {"leaderboard/table": table}

    def _bar(metric: str, title: str) -> None:
        rows = [[r.get("model"), r.get(metric)] for r in data.summary_rows
                if isinstance(r.get(metric), num)]
        if rows:
            t = wandb.Table(data=rows, columns=["model", metric])
            panels[f"leaderboard/{metric}"] = wandb.plot.bar(t, "model", metric, title=title)

    _bar("best_tps", "Decode TPS (HF eager, bs=1)")
    _bar("peak_vram_mb", "Peak VRAM (MB, HF)")
    _bar("ppl", "Perplexity (wikitext-2, lower=better)")
    _bar("mmlu_acc", "MMLU-tiny accuracy")
    _bar("tpot_ms_p50", "TPOT p50 (ms, lower=better)")

    vllm_rows = []
    for r in data.summary_rows:
        s = _serving_for(r.get("model", ""), serving)
        if s and s.get("loaded") and isinstance(s.get("decode_tps"), num):
            vllm_rows.append([r.get("model"), s["decode_tps"]])
    if vllm_rows:
        t = wandb.Table(data=vllm_rows, columns=["model", "decode_tps_vllm"])
        panels["leaderboard/decode_tps_vllm"] = wandb.plot.bar(
            t, "model", "decode_tps_vllm", title="Decode TPS (vLLM, single stream)")

    hf_vs_vllm = []
    for r in data.summary_rows:
        s = _serving_for(r.get("model", ""), serving)
        if isinstance(r.get("best_tps"), num) and s and isinstance(s.get("decode_tps"), num):
            hf_vs_vllm.append([r.get("model"), "hf_eager", r["best_tps"]])
            hf_vs_vllm.append([r.get("model"), "vllm", s["decode_tps"]])
    if hf_vs_vllm:
        t = wandb.Table(data=hf_vs_vllm, columns=["model", "runtime", "decode_tps"])
        panels["leaderboard/hf_vs_vllm_table"] = t

    run.log(panels)
    print(f"Leaderboard run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
