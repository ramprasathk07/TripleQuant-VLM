"""'BEST FOR ...' verdicts computed from a comparison_summary's flattened rows.

This is the folded-in "advisor" from the roadmap: rather than a separate rules
module, the report generator derives BEST FOR LATENCY/MEMORY/QUALITY/BALANCED
directly from the same numbers in the table above it. All logic here is a
documented heuristic (min-max normalized average for "balanced"), not a claim
of some deeper multi-objective optimum -- the report labels it as such.
"""
from __future__ import annotations

_NUM = (int, float)


def _best(rows: list[dict], key: str, minimize: bool) -> dict | None:
    candidates = [r for r in rows if isinstance(r.get(key), _NUM)]
    if not candidates:
        return None
    fn = min if minimize else max
    return fn(candidates, key=lambda r: r[key])


def _minmax_norm(values: list[float], minimize: bool) -> list[float]:
    """Map each value to [0, 1] where 1 is always 'best'."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    if minimize:
        return [(hi - v) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


def _balanced_verdict(rows: list[dict]) -> dict | None:
    quality_key = "mmlu_acc" if any(isinstance(r.get("mmlu_acc"), _NUM) for r in rows) else "ppl"
    quality_minimize = quality_key == "ppl"

    candidates = [
        r for r in rows
        if isinstance(r.get("tpot_ms_p50"), _NUM)
        and isinstance(r.get("peak_vram_mb"), _NUM)
        and isinstance(r.get(quality_key), _NUM)
    ]
    if len(candidates) < 2:
        return None   # nothing meaningful to normalize against

    lat_scores = _minmax_norm([r["tpot_ms_p50"] for r in candidates], minimize=True)
    mem_scores = _minmax_norm([r["peak_vram_mb"] for r in candidates], minimize=True)
    q_scores   = _minmax_norm([r[quality_key] for r in candidates], minimize=quality_minimize)
    avg = [(l + m + q) / 3 for l, m, q in zip(lat_scores, mem_scores, q_scores)]

    best_i = max(range(len(candidates)), key=lambda i: avg[i])
    return {
        "model": candidates[best_i]["model"],
        "score": round(avg[best_i], 3),
        "quality_metric": quality_key,
        "note": "equal-weighted average of normalized TPOT/VRAM/" + quality_key + " across models with all three metrics",
    }


def compute_verdicts(rows: list[dict]) -> dict:
    """BEST FOR LATENCY / MEMORY / QUALITY / BALANCED from comparison-table rows.

    Only 'success' rows are considered. Any verdict whose underlying metric is
    absent from every row is simply omitted (never a fabricated placeholder).
    """
    ok_rows = [r for r in rows if r.get("status") == "success"]
    verdicts: dict = {}

    lat_best = _best(ok_rows, "tpot_ms_p50", minimize=True)
    if lat_best:
        verdicts["latency"] = {
            "model": lat_best["model"], "metric": "tpot_ms_p50", "value": lat_best["tpot_ms_p50"],
        }

    mem_best = _best(ok_rows, "peak_vram_mb", minimize=True)
    if mem_best:
        verdicts["memory"] = {
            "model": mem_best["model"], "metric": "peak_vram_mb", "value": mem_best["peak_vram_mb"],
        }

    if any(isinstance(r.get("mmlu_acc"), _NUM) for r in ok_rows):
        q_best = _best(ok_rows, "mmlu_acc", minimize=False)
        if q_best:
            verdicts["quality"] = {
                "model": q_best["model"], "metric": "mmlu_acc", "value": q_best["mmlu_acc"],
            }
    elif any(isinstance(r.get("ppl"), _NUM) for r in ok_rows):
        q_best = _best(ok_rows, "ppl", minimize=True)
        if q_best:
            verdicts["quality"] = {
                "model": q_best["model"], "metric": "ppl", "value": q_best["ppl"],
            }

    balanced = _balanced_verdict(ok_rows)
    if balanced:
        verdicts["balanced"] = balanced

    return verdicts
