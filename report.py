#!/usr/bin/env python
"""Main entry point for turning benchmark results into a human-readable report.

Reads the latest (or an explicit) comparison_summary_*.json from a results
directory plus its matching per-model detail JSONs, and writes report.md +
plots/ + summary.json — a leaderboard table, BEST-FOR verdicts, and charts
(latency/memory/quality bars, VRAM-vs-context, TurboQuant bit-width tradeoff)
built from whichever of those the run actually collected.

Usage:
    python report.py --dir results/qwen3-1.7b-sweep
    python report.py --dir results/qwen3-1.7b-sweep --summary results/qwen3-1.7b-sweep/comparison_summary_20260717T150027Z.json
    python report.py --dir results/qwen3-1.7b-sweep --out docs/benchmark_report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils import setup_global_logging

logger = setup_global_logging("report")

from src.reporting.generate import generate_report


def main() -> None:
    """Parse arguments, generate the report, and exit 1 if the results dir is missing or empty."""
    parser = argparse.ArgumentParser(
        description="TripleQuant-VLM — turn benchmark results into report.md + plots/.",
    )
    parser.add_argument(
        "--dir", "-d", required=True,
        help="results/<run_name>/ directory produced by benchmark.py",
    )
    parser.add_argument(
        "--summary", default=None,
        help="Explicit comparison_summary_*.json (default: latest one in --dir)",
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Output directory (default: <dir>/report)",
    )
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    try:
        report_path = generate_report(
            results_dir,
            summary_path=Path(args.summary) if args.summary else None,
            out_dir=Path(args.out) if args.out else None,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Report written -> %s", report_path)


if __name__ == "__main__":
    main()
