# Experiment History

Chronological log of what each benchmark sweep in `results/` actually tested and found.
Raw JSON lives in `results/<run_name>/` (gitignored — regenerate with `benchmark.py`, or
ask for the archive). This log is the human-readable index into that history.

---

## 2026-05-27 to 2026-05-29 — Pipeline validation

**Goal:** confirm the dual-runtime benchmark harness (`benchmark.py`) produces real
numbers end to end, not just `--dry-run` validation.

- `config/benchmark/llm_comparison.yaml` — text LLM path (PPL + MMLU-tiny + TTFT/TPOT +
  throughput + memory) smoke-tested on the HF runtime.
- `config/benchmark/ocr_comparison.yaml` — VLM/OCR path (CER/WER/EM/BLEU) smoke-tested.
- *(Both generic configs were later removed in the post-v1.0.0 cleanup — they pointed at
  since-deleted TinyLlama-era checkpoint paths and were superseded by the per-model
  configs. Recoverable from git history.)*
- **Qwen2.5-VL-3B AWQ-W4A16 OCR sweep** (`results/qwen2.5-vl-3b-ocr-sweep/`, 5 runs,
  2026-05-29): 4 of 5 succeeded with identical CER = 0.1733 (deterministic eval, as
  expected at temperature 0 / fixed seed); the 5th run failed (mid-iteration on an
  unrelated change, not investigated further at the time — OCR path was already proven
  working by the other 4).

**Outcome:** harness works. Both quality-metric families (text logit-based, VLM
generation-based) produce stable, reproducible numbers.

---

## 2026-06-06 to 2026-06-08 — First TurboQuant sweep + tracking wiring

**Goal:** get Qwen3-1.7B fp16-baseline vs TurboQuant K3V2 running side by side and wire
up W&B/TensorBoard tracking.

- 8 sweep iterations across the two days (`results/qwen3-1.7b-sweep/`,
  `comparison_summary_20260606T062214Z.json` through `...20260608T055831Z.json`) — mostly
  re-runs while iterating on the tracking integration, not distinct experiments.
- The **2026-06-08T055831Z** run is the first with `ctx_sweep` and `tq_bits_sweep`
  populated (both metrics already existed in `hf_runtime.py` at this point — they just
  weren't being surfaced anywhere until the report generator, M2, was built over a month
  later). This run first showed:
  - K3V2 next-token agreement 9-19% vs FP16 (the low-bit quality tradeoff, later confirmed
    reproducible — see `failure_cases.md` #2).
  - fp16's `ctx_sweep` reaching `peak_vram_mb: 14174` at 8192 tokens on a 12GB card — at
    the time this read as "TurboQuant barely wins on memory." It was actually the Windows
    VRAM-oversubscription bug (`failure_cases.md` #1), not caught until this run's data
    was re-examined on 2026-07-17.
- PPL and MMLU-tiny were bit-identical between fp16 and TurboQuant in every run this
  period (`22.449370186003648` to 16 decimal places) — see `failure_cases.md` #4 for why
  (teacher-forced eval never reads the KV cache).

**Outcome:** TurboQuant is wired into the decode path and producing real, differentiated
numbers in `tq_bits_sweep`/`ctx_sweep` — but nobody was looking at those fields yet (no
report generator existed), and the one field that *was* being read (`peak_vram_mb` at
short context) understated the story while also being quietly corrupted by the
oversubscription bug.

---

## 2026-07-17 — torchao backend, VRAM bug fix, corrected sweep

**Goal:** land the torchao quantization backend (M0 of `notes/plan_v1.0.md`), then make
TurboQuant's value measurable (M1) — which meant first re-examining the June data closely
enough to find the bugs above.

1. **torchao backend landed** (`src/quantization/torch_ao.py` + on-the-fly
   `HFRuntime._apply_torchao`) — offline quantize -> save -> reload -> generate verified
   on Qwen3-1.7B; benchmark path verified with the new `model_vram_mb` metric.
2. **VRAM oversubscription bug found** while sanity-checking the M0 benchmark numbers
   against the June data (`14174 MB` peak on a `12288 MB` card doesn't parse). Root cause
   and fix in `failure_cases.md` #1. Verified independently via a direct
   `measure_ctx_sweep`/`measure_max_context` call before trusting it in the real sweep.
3. **Environment metadata capture added** (`benchmark.py: _capture_environment`) —
   reproducibility gap flagged in the original M2 plan, closed proactively; every sweep
   from this commit onward records GPU/driver/CUDA/torch/transformers/git-commit.
4. **Fresh 3-model sweep** (`comparison_summary_20260717T152125Z.json`): fp16-baseline,
   torchao-int8wo, turboquant-k3v2, all on the corrected measurement code. This is the
   sweep `docs/benchmark_report.md` is built from. Total wall-clock: ~20 minutes for all
   three models — markedly faster than any of the June runs, because the VRAM fix removes
   the WDDM-paging slowdown that used to make the fp16 baseline's context sweep crawl at
   PCIe speed near its (fake) ceiling.
5. **Report generator built** (`src/reporting/` + `report.py`, M2) — the first tool able
   to actually surface `ctx_sweep`/`tq_bits_sweep` in a human-readable form. Verified
   against both the July 17 fresh data and the June 8 historical data (confirming the
   K3V2 tradeoff shape is stable across a month-long gap, not a one-off).

**Outcome:** the corrected context-length sweep shows TurboQuant K3V2 reaching 16,384
usable tokens vs fp16's 4,096 on the same 12GB card (4x) — the actual, trustworthy
headline result, now backed by a report a stranger can read.

---

## Reproducing any of these

```bash
python benchmark.py -c config/benchmark/qwen3_1_7b.yaml    # re-run the July 17 sweep
python report.py --dir results/qwen3-1.7b-sweep            # regenerate the report
```

Every sweep from 2026-07-17 onward embeds its own environment snapshot in the result
JSON — no need to cross-reference this log to know what hardware/software produced a
given number going forward.
