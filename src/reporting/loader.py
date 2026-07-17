"""Load a benchmark comparison_summary + its per-model detail JSONs from disk."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReportData:
    run_name: str | None
    timestamp: str
    environment: dict | None
    summary_rows: list[dict]
    detail_records: dict[str, dict] = field(default_factory=dict)   # model_name -> full record
    results_dir: Path | None = None
    summary_path: Path | None = None


def find_latest_summary(results_dir: Path) -> Path:
    """Most recent comparison_summary_*.json in *results_dir*.

    Filenames embed a UTC timestamp as YYYYMMDDTHHMMSSZ, so a lexicographic sort
    is also a chronological sort -- no need to parse the timestamp out.
    """
    candidates = sorted(results_dir.glob("comparison_summary_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No comparison_summary_*.json found in {results_dir}. "
            "Run benchmark.py against a config first."
        )
    return candidates[-1]


def load_report_data(results_dir: Path, summary_path: Path | None = None) -> ReportData:
    """Load one benchmark run's summary plus the detail JSON for every model in it.

    Detail files share the summary's run timestamp in their filename
    (``{model}_{runtime}_{run_ts}.json`` vs ``comparison_summary_{run_ts}.json``),
    which is how the two are matched up.
    """
    results_dir = Path(results_dir)
    summary_path = Path(summary_path) if summary_path else find_latest_summary(results_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    run_ts = summary_path.stem.replace("comparison_summary_", "")
    detail_records: dict[str, dict] = {}
    for f in results_dir.glob(f"*_{run_ts}.json"):
        if f.name.startswith("comparison_summary_"):
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        detail_records[rec.get("model_name", f.stem)] = rec

    return ReportData(
        run_name=summary.get("run_name"),
        timestamp=summary.get("timestamp", ""),
        environment=summary.get("environment"),
        summary_rows=summary.get("records", []),
        detail_records=detail_records,
        results_dir=results_dir,
        summary_path=summary_path,
    )
