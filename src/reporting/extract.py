"""Shared extraction of scalar/table views from a per-model metrics dict.

Pulled out of benchmark.py so both the live benchmark run (W&B / TensorBoard
logging) and the offline report generator (src/reporting/generate.py) read the
same nested metrics dict the same way. The metrics dict itself is whatever a
"metrics" key in a results/*.json record holds -- {"memory": {...},
"quality_llm": {...}, "quality_ocr": {...}, "perf": {...}}.
"""
from __future__ import annotations


def flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    """Flatten a nested benchmark metrics dict into scalar `key -> number` pairs.

    - dicts recurse with a "/"-joined prefix;
    - a throughput list (items with `batch_size`) expands to `.../bs{n}/...`;
    - booleans become 0/1; non-numeric values (strings, None, error/oom markers)
      are dropped so only loggable scalars remain.
    """
    out: dict = {}
    for key, val in metrics.items():
        if key in ("per_sample", "error", "detail"):
            continue
        full = f"{prefix}{key}"
        if isinstance(val, bool):
            out[full] = int(val)
        elif isinstance(val, (int, float)):
            out[full] = val
        elif isinstance(val, dict):
            out.update(flatten_metrics(val, prefix=f"{full}/"))
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            for item in val:
                tag = f"bs{item['batch_size']}" if "batch_size" in item else None
                if tag is None:
                    continue
                out.update(flatten_metrics(
                    {k: v for k, v in item.items() if k != "batch_size"},
                    prefix=f"{full}/{tag}/",
                ))
    return out


def ctx_rows(metrics: dict) -> list:
    """Series points that are a genuine fit — excludes the point (if any) where
    WDDM paged into shared system memory (oversubscribed=True); that point stays
    in the raw JSON for transparency but isn't a real capacity result."""
    sweep = metrics.get("perf", {}).get("ctx_sweep", {})
    series = sweep.get("series", []) if isinstance(sweep, dict) else []
    return [s for s in series if s.get("fits") and not s.get("oversubscribed")]


def bits_grid(metrics: dict) -> list:
    sweep = metrics.get("perf", {}).get("tq_bits_sweep", {})
    return sweep.get("grid", []) if isinstance(sweep, dict) else []


def tput_rows(metrics: dict) -> list:
    tput = metrics.get("perf", {}).get("throughput", [])
    return [r for r in tput if isinstance(r, dict) and not r.get("oom")] if isinstance(tput, list) else []


def curated_scalars(metrics: dict) -> dict:
    """Headline perf / memory / quality scalars to log as comparison panels.

    A curated subset (not every nested value) so the W&B workspace shows clean,
    cross-run bar panels instead of dozens of single-point charts.
    """
    out: dict = {}
    num = (int, float)

    mem = metrics.get("memory", {})
    if isinstance(mem, dict):
        for k in ("peak_vram_mb", "model_vram_mb", "disk_mb", "load_time_s"):
            if isinstance(mem.get(k), num):
                out[f"memory/{k}"] = mem[k]

    perf = metrics.get("perf", {})
    if isinstance(perf, dict):
        lat = perf.get("latency", {})
        if isinstance(lat, dict):
            for k in ("ttft_ms_p50", "ttft_ms_p99", "tpot_ms_p50", "tpot_ms_p99"):
                if isinstance(lat.get(k), num):
                    out[f"latency/{k}"] = lat[k]
        tps = [r.get("tokens_per_sec", 0) for r in tput_rows(metrics)]
        if tps:
            out["throughput/best_tokens_per_sec"] = max(tps)
        mc = perf.get("max_context", {})
        if isinstance(mc, dict):
            for k in ("measured_max_tokens", "model_max_positions"):
                if isinstance(mc.get(k), num):
                    out[f"max_context/{k}"] = mc[k]
        cs = perf.get("ctx_sweep", {})
        if isinstance(cs, dict) and isinstance(cs.get("est_max_ctx_tokens"), num):
            out["max_context/est_max_ctx_tokens"] = cs["est_max_ctx_tokens"]

    q = metrics.get("quality_llm", {})
    if isinstance(q, dict):
        if isinstance(q.get("ppl"), num):
            out["quality/ppl"] = q["ppl"]
        if isinstance(q.get("mmlu_acc"), num):
            out["quality/mmlu_acc"] = q["mmlu_acc"]
        for name in ("gsm8k", "aime", "ceval"):
            v = q.get(name)
            if isinstance(v, dict) and isinstance(v.get("acc"), num):
                out[f"quality/{name}_acc"] = v["acc"]
    qo = metrics.get("quality_ocr", {})
    if isinstance(qo, dict):
        for k in ("cer", "wer", "exact_match", "bleu"):
            if isinstance(qo.get(k), num):
                out[f"quality/{k}"] = qo[k]
    return out
