"""Render report.md from loaded results + computed verdicts + generated plots."""
from __future__ import annotations

from .extract import bits_grid
from .loader import ReportData

_NUM = (int, float)

# (row key, header, value format)
_OPTIONAL_COLS = [
    ("ttft_ms_p50",  "TTFT (ms)",  ".1f"),
    ("tpot_ms_p50",  "TPOT (ms)",  ".1f"),
    ("best_tps",     "TPS",        ".1f"),
    ("peak_vram_mb", "VRAM (MB)",  ".1f"),
    ("disk_mb",      "Disk (MB)",  ".1f"),
    ("ppl",          "PPL",        ".2f"),
    ("mmlu_acc",     "MMLU",       ".3f"),
    ("ocr_cer",      "CER",        ".3f"),
]


def _fmt(v, fmt: str) -> str:
    return f"{v:{fmt}}" if isinstance(v, _NUM) else "—"


def _leaderboard_table(rows: list[dict]) -> str:
    if not rows:
        return "_No results in this summary._\n"
    shown = [c for c in _OPTIONAL_COLS if any(isinstance(r.get(c[0]), _NUM) for r in rows)]
    headers = ["Model", "Runtime", "Status"] + [h for _, h, _ in shown]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [str(r.get("model", "—")), str(r.get("runtime", "—")), str(r.get("status", "—"))]
        cells += [_fmt(r.get(key), fmt) for key, _, fmt in shown]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _environment_block(env: dict | None) -> str:
    if not env:
        return "_Environment metadata not recorded for this run._\n"
    fields = [
        ("GPU", env.get("gpu_name")),
        ("VRAM", f"{env['gpu_total_vram_mb']:.0f} MB" if isinstance(env.get("gpu_total_vram_mb"), _NUM) else None),
        ("Driver", env.get("gpu_driver_version")),
        ("CUDA", env.get("cuda_version")),
        ("Torch", env.get("torch_version")),
        ("Transformers", env.get("transformers_version")),
        ("Python", env.get("python_version")),
        ("OS", env.get("os")),
        ("Commit", env.get("git_commit")),
    ]
    return "\n".join(f"- **{k}:** {v}" for k, v in fields if v) + "\n"


def _fmt_num(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def _verdicts_block(verdicts: dict) -> str:
    if not verdicts:
        return "_Not enough successful results to compute verdicts._\n"
    lines = []
    if "latency" in verdicts:
        v = verdicts["latency"]
        lines.append(f"- **Best for latency:** `{v['model']}` — {v['metric']} = {_fmt_num(v['value'])}")
    if "memory" in verdicts:
        v = verdicts["memory"]
        lines.append(f"- **Best for memory:** `{v['model']}` — {v['metric']} = {_fmt_num(v['value'])}")
    if "quality" in verdicts:
        v = verdicts["quality"]
        lines.append(f"- **Best for quality:** `{v['model']}` — {v['metric']} = {_fmt_num(v['value'])}")
    if "balanced" in verdicts:
        v = verdicts["balanced"]
        lines.append(f"- **Best balanced:** `{v['model']}` — score {v['score']} _({v['note']})_")
    return "\n".join(lines) + "\n"


def _tq_bits_table(detail_records: dict[str, dict]) -> str | None:
    grid: list[dict] = []
    for rec in detail_records.values():
        grid.extend(bits_grid(rec.get("metrics", {})))
    if not grid:
        return None
    headers = ["Bits", "Context", "Top-1 agree", "Top-5 agree", "KL div", "KV MB", "Compression"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in sorted(grid, key=lambda g: (g["key_bits"] + g["value_bits"], g["context_len"])):
        lines.append(
            f"| {r['bits']} | {r['context_len']} | {r['top1_agreement']:.3f} | "
            f"{r.get('top5_agreement', 0):.3f} | {r['kl_div']:.4f} | "
            f"{r['kv_cache_mb']:.1f} | {r.get('compression_ratio', 1.0):.2f}x |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(data: ReportData, verdicts: dict, plot_paths: dict[str, str]) -> str:
    parts: list[str] = []
    title = data.run_name or data.results_dir.name if data.results_dir else "Benchmark Report"
    parts.append(f"# {title}\n")
    parts.append(f"_Generated from `{data.summary_path.name}` (run timestamp: {data.timestamp})_\n")

    parts.append("## Environment\n")
    parts.append(_environment_block(data.environment))

    parts.append("## Results\n")
    parts.append(_leaderboard_table(data.summary_rows))

    parts.append("## Verdicts\n")
    parts.append(_verdicts_block(verdicts))

    if "latency" in plot_paths:
        parts.append("## Latency\n")
        parts.append(f"![Latency]({plot_paths['latency']})\n")
    if "memory" in plot_paths:
        parts.append("## Memory\n")
        parts.append(f"![Memory]({plot_paths['memory']})\n")
    if "quality" in plot_paths:
        parts.append("## Quality\n")
        parts.append(f"![Quality]({plot_paths['quality']})\n")
    if "ctx_sweep" in plot_paths:
        parts.append("## VRAM vs Context Length\n")
        parts.append(f"![VRAM vs context length]({plot_paths['ctx_sweep']})\n")
    if "tq_bits" in plot_paths:
        parts.append("## TurboQuant: Compression vs Quality Tradeoff\n")
        parts.append(
            "Next-token agreement with the FP16 baseline, per KV bit-width, at the "
            "compressed-tail positions (the ring buffer's most recent tokens stay "
            "exact regardless of bit-width).\n"
        )
        parts.append(f"![TurboQuant bits tradeoff]({plot_paths['tq_bits']})\n")
        tbl = _tq_bits_table(data.detail_records)
        if tbl:
            parts.append(tbl)

    parts.append("---\n")
    parts.append(f"_Source: `{data.results_dir}` — {len(data.detail_records)} detail record(s) matched._\n")

    return "\n".join(parts)
