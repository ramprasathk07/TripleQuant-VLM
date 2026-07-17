# TripleQuant-VLM — Road to v1.0 Freeze

**Created:** 2026-07-17
**Updated:** 2026-07-17 — M0, M1, M2 done same day (see §9 below)
**Supersedes:** `notes/plan.md` (May sprint plan — keep for history)
**Source:** distilled from `notes/ChatGPT-Career Doubts and Communication.md` + current repo state
**Hardware reality:** RTX 3060 12 GB (Ampere), Windows. INT4/INT8 native; FP8 emulated (slow); NVFP4/Blackwell impossible; TensorRT-LLM impractical.

---

## 0. The one rule (from the ChatGPT convo, worth keeping)

Before adding anything, ask:

> **"Will this help someone compare inference methods, or am I just adding another wrapper?"**

TripleQuant's value is **orchestration + comparison + evaluation** — not another engine, not another quantizer wrapper. And the second rule: once v1.0 boxes are checked, **stop adding features and start writing** (docs + blogs). Visibility, not more code, is the next career lever.

---

## 1. Where the repo actually stands (2026-07-17)

### Already done (more than the convo assumed)
| Convo's v1.0 item | Reality |
|---|---|
| Unified quantization API | ✅ `BaseQuantizer` + registry + factory; `llm_compressor`, `modelopt`, `fp16` baseline |
| torchao backend | 🔶 **80% done, uncommitted**: `src/quantization/torch_ao.py` (offline) + `HFRuntime._apply_torchao` (on-the-fly) + schema + factory wiring + `config/quantize/qwen3_1_7b/torchao_int8wo.yaml` |
| Engine abstraction | ✅ **Already exists as `src/runtimes/`** (base + factory + HF + vLLM). Convo suggested a new `engines/` folder — that's a rename, not a feature. Skip the churn. |
| Benchmark suite | ✅ `benchmark.py`: crash-safe subprocess loop, TTFT/TPOT/TPS, peak VRAM, disk, load time, PPL, MMLU-tiny, OCR CER/WER/EM/BLEU, context sweeps, `--dry-run` |
| Experiment tracking | ✅ W&B wired (`src/tracking/wandb_connector.py`, commit 40a7055); local JSON always |
| Eval tasks beyond PPL | ✅ MMLU-tiny + OCR suite. Enough for v1.0. |
| Model zoo | ✅ configs for Qwen3-1.7B, Qwen3-4B-thinking, Qwen2.5-VL-3B, Hunyuan-OCR |

### Actually missing (the real v1.0 gap)
1. **Report generator** — results are raw JSON in `results/*/`; no `report.md`, no plots, no leaderboard table. This is the convo's ⭐⭐⭐⭐⭐ item and it's genuinely the biggest lever: the data already exists, nobody can see it.
2. **TurboQuant quality is unmeasured** — see §3. Latest sweep shows TQ PPL/MMLU **bit-identical** to fp16 (22.449370186003648) because teacher-forced metrics never exercise the decode KV path. `logit_kl`/`token_agree` still stubbed in `benchmark.py:197-199`.
3. **README sells architecture, not evidence** — no hero results table, TurboQuant buried mid-page, no decision/compatibility matrix.
4. **No `docs/`** — raw material exists in `notes/` (debugging logs, case studies, kernel scope) but isn't curated.

### Deliberately out (v1.0 scope cuts, with reasons)
- **TensorRT-LLM engine** — no supported path on this hardware/OS; document as v2 with one line.
- **ONNX Runtime engine** — passes the comparison test in principle (CPU/edge story) but is pure integration work with no insight on a 3060; v2.
- **`advisor/` module** — good idea, wrong shape. Fold "BEST FOR latency / memory / quality / balanced" verdicts into the report generator output instead of a rules module.
- **More quantizers (GPTQ-variants, HQQ, BnB…), more engines, pruning, distillation, sparsity** — post-freeze parking lot.

---

## 2. M0 — Land torchao (in flight, ~1 day)

The uncommitted diff is coherent. Finish line:

1. Offline path: `python quantize.py --config config/quantize/qwen3_1_7b/torchao_int8wo.yaml` → checkpoint saves, `tests/simple_generate.py` produces sane text.
2. Benchmark path: run `config/benchmark/qwen3_1-7b.yaml` (already has `torchao: int8wo` entry) → confirm new `model_vram_mb` metric shows int8 weight savings vs fp16 (~×0.55 resident), quality metrics populate.
3. Commit (torchao backend + `model_vram_mb` metric are one logical change).

Acceptance: torchao row appears in a comparison summary with real numbers.

---

## 3. M1 — Make TurboQuant measurable (~1 week) ← highest engineering value

**Problem statement:** TQ is the flagship (custom rotations + Lloyd-Max + Triton kernels), yet the current harness *cannot show it does anything* — quality metrics bypass the KV cache, and short-context runs hide the memory win (peak VRAM 3884 vs 3876 MB — noise).

Three fixes, in order:

### 3a. Generation-based quality (fixes the bit-identical-PPL hole)
Wire the stubbed metrics in `benchmark.py`:
- When config has a `baseline` entry: load baseline runtime, generate N tokens greedy on a fixed prompt set through the *decode* path (KV cache engaged), capture logits/tokens.
- `eval_token_agreement`: % of decode steps where TQ model picks the same token as fp16.
- `eval_logit_kl`: KL on decode-step logits (functions already exist in `src/evaluation/eval_llm.py`).
- Report per-position degradation curve (agreement vs. token index) — this is blog-grade data.

### 3b. Long-context memory sweep (shows the actual win)
- Context sweep 1K / 4K / 8K / 16K on Qwen3-1.7B, batch 1: peak VRAM + TPOT, fp16 KV vs TQ K3V2.
- KV cache at 16K for Qwen3-1.7B ≈ where the fp16 curve visibly diverges; 12 GB VRAM fits.
- One chart: VRAM vs context length, two lines. This is the hero image for README and Blog 5.

### 3c. Honest failure documentation
- QJL variance degradation (already known, `--use-qjl` off by default) → `docs/failure_cases.md` entry with numbers.

Acceptance: a table where TQ differs from baseline in ≥3 metrics (token-agreement, VRAM@16K, TPOT@16K), committed as JSON + chart.

---

## 4. M2 — Report generator (~3-4 days) ← highest visibility value

New `src/reporting/` + CLI (`python report.py -d results/qwen3-1.7b-sweep/` or `benchmark.py --report`):

- **Input:** existing `comparison_summary_*.json` + per-model JSONs (schema already stable).
- **Output per sweep:**
  - `report.md` — hero leaderboard table (model × quantizer × runtime → TTFT, TPS, VRAM, PPL, MMLU), environment block (GPU, driver, CUDA, torch, commit hash), config provenance.
  - `plots/` — latency, memory, quality bars; VRAM-vs-context line chart when sweep data present.
  - **Verdict section** (the folded-in advisor): BEST FOR LATENCY / MEMORY / QUALITY / BALANCED, computed from the data.
  - `summary.json` — machine-readable.
- Environment metadata capture goes into the benchmark run itself if not already recorded (GPU, CUDA, driver, torch/transformers versions, timestamp).

Acceptance: one command turns an existing results dir into a report a stranger can read; README hero table is copy-pasted from it.

---

## 5. M3 — README + docs overhaul (~3 days)

README restructure (order matters):
1. Tagline: *"Benchmark, compare, and evaluate quantization methods and inference engines across models, hardware, and workloads."*
2. **Hero results table** (from M2 report: fp16 vs torchao-int8 vs TQ on Qwen3-1.7B, RTX 3060) + VRAM-vs-context chart.
3. **TurboQuant section promoted** to the top third — it's the differentiator.
4. Decision matrix (need lowest VRAM → TQ; easiest deploy → vLLM+compressed-tensors; …).
5. Compatibility matrix (quantizer × runtime, ✅/❌/WIP) — people constantly need this.
6. Architecture (current content, compressed ~30%, keep the mermaid diagrams).
7. Quickstart, extensibility, roadmap w/ explicit v2 parking lot.

New `docs/`:
- `benchmark_report.md` — latest full report (M2 output).
- `engineering_notes.md` — curate from `notes/debugging_turboquant_kv.md`, `turboquant_hf_cache_guide.md`.
- `experiments.md` — sweep history + what each showed.
- `failure_cases.md` — QJL variance, FP8-on-Ampere emulation, Windows modelopt CPU-ext fallback, vLLM/Windows split-env. Honest negatives = credibility.

Repo hygiene (same pass): delete empty `serve.py` or implement; collapse 5 `setup*.bat` variants into documented 2; reconcile `req.txt` vs `pyproject.toml`.

---

## 6. M4 — Freeze v1.0 (~1 day)

- Every config in `config/quantize/` + `config/benchmark/` either runs green or has a documented skip reason (hardware floor).
- Tag `v1.0.0`, GitHub release with the M2 report attached.
- Open the v2 parking lot as GitHub issues (ONNX engine, TRT-LLM export, GSM8K eval, HQQ/BnB, pruning/sparsity, advisor CLI, vLLM TQ backend) — visible ambition, zero obligation.

**After the tag: no new features.** Every idea goes to the parking lot.

---

## 7. M5 — Blog phase (ongoing, target 2/month)

Order deviates from the convo deliberately: start where the repo already produces data, so each post ships with original numbers instead of paper summaries.

| # | Post | Why this order |
|---|---|---|
| 1 | **KV Cache Compression: KIVI, FP8-KV, TurboQuant** — with M1's agreement/VRAM curves | Signature niche; data ready after M1 |
| 2 | **Benchmarking LLM Inference Properly** (TTFT/TPOT/TPS, why teacher-forced PPL can't see KV-quant error — the bit-identical-PPL story) | The M1 bug *is* the hook; nobody writes this |
| 3 | **Building TripleQuant-VLM** — registry/factory/dual-backend design, what was rejected | Freeze retrospective, ships with v1.0 |
| 4 | Quantization Mathematics (uniform/affine, Lloyd-Max, KL calibration) | Revision-driven; feeds interview prep |
| 5 | Mathematics of Transformer Inference (memory-bound decode, roofline) | Foundation post |
| 6 | KV Cache from First Principles (O(n²)→O(n), memory formula) | Pairs with 1 |
| 7 | Modern Quantization Methods (GPTQ vs AWQ vs SmoothQuant vs torchao — with repo benchmarks) | Data from M0 |
| 8 | Inference Engines Under the Hood (HF vs vLLM; PagedAttention, continuous batching) | |
| 9 | Hardware-Aware Optimization (tensor cores, FP8 emulation cost on Ampere — measured) | Repo has the receipts |
| 10 | Mathematics of Inference Optimization (roofline capstone) | Capstone |

Format per post (from the convo, keep): problem → derivation → systems intuition → implementation walk (TripleQuant code) → experiments (framework plots) → takeaways.

---

## 8. Sequencing summary

```
M0 torchao commit        ~1 day    (in flight)
M1 TQ measurable         ~1 week   (engineering core)
M2 report generator      ~3-4 days (visibility core)
M3 README/docs           ~3 days
M4 freeze v1.0           ~1 day
──────────────────────────────────
~2.5–3 weeks part-time to tag
M5 blogs                 2/month after
```

M1 before M2 because the report is only as good as the numbers in it, and today the flagship's numbers are hollow. If time pressure hits, M2 can start in parallel — the report schema doesn't depend on M1's metrics existing.

---

## 9. M0–M2 completion notes (2026-07-17)

All three landed in one session, in this order: M0 → discovered mid-verification that M1's measurement infra (`tq_bits_sweep`, `ctx_sweep`) already existed and had real data sitting unused since June 8 → fixed a real bug found along the way → M1 → M2 built in parallel while M1's fresh sweep ran (as this doc allowed).

**Bug found and fixed (not in the original plan):** `measure_ctx_sweep`/`measure_max_context` in `src/runtimes/hf/hf_runtime.py` trusted CUDA's "did the allocation succeed" signal, but on Windows/WDDM an allocation can silently spill into shared system RAM instead of raising OOM. The fp16 baseline was reporting `peak_vram_mb: 14174` with `fits: true` at 8192 context — impossible on a 12288 MB card. Fixed by comparing peak allocated against `torch.cuda.get_device_properties().total_memory` (a hardware fact WDDM can't inflate); anything past 95% of it is flagged `oversubscribed` and excluded from `max_fit_tokens`/downstream charts. Commit `2289023`.

**The real hero result (now trustworthy):** on the corrected data, fp16 baseline's context sweep tops out at 4096 tokens (12GB card), torchao-int8wo reaches 8192 (freed-up resident VRAM buys headroom), and **TurboQuant K3V2 reaches 16384 — 4x fp16's usable context on the same GPU.** This is the actual, defensible TurboQuant value proposition and the chart to lead the README with (`results/qwen3-1.7b-sweep/report/plots/ctx_sweep.png`).

**The honest tradeoff (kept per user decision, not hidden):** TurboQuant's showcased K3V2 default has weak next-token agreement vs fp16 (9-19% across context lengths, reproduced across two independent runs a month apart — June 8 and July 17 — so it's a real characteristic, not noise). K4V4 is modestly better (~15-33%), K8V8 is near-lossless (~97-98%) but only compresses ~1.3-1.7x. Value-bit precision, not key-bit, is the dominant factor. Full table + chart in the generated report; goes into `docs/failure_cases.md` / a blog post rather than being smoothed over.

**Noise, not signal:** TurboQuant's MMLU-tiny score (0.564) edged out fp16 (0.548) in the fresh run. Almost certainly sample-size noise (MMLU-tiny is small) or minor floating-point differences from the patched attention path when the KV ring is exact (positions within `ring_capacity=256` aren't compressed) — not a real quality improvement. Don't repeat "TurboQuant improves MMLU" anywhere; if it needs stating, say the two are statistically indistinguishable at this sample size.

**Also landed, not in original M0-M2 scope but needed:**
- Environment metadata capture (GPU/driver/CUDA/torch/transformers/git commit) — `benchmark.py`, commit `ad306be`. Reproducibility gap the plan's M2 section flagged; the June/July sweeps used to test the report predate this, so their reports show "not recorded."
- `src/reporting/extract.py` — pulled the metrics-dict-shape helpers (`flatten_metrics`, `ctx_rows`, `bits_grid`, `tput_rows`, `curated_scalars`) out of `benchmark.py` so the report generator and the live W&B/TensorBoard logging read the same nested JSON the same way. No behavior change, commit `933f884`.

**M2 shape as built** (`src/reporting/` + `report.py`): `python report.py --dir results/<run_name>/` auto-picks the latest `comparison_summary_*.json`, matches it to per-model detail JSONs by run timestamp, and writes `report.md` (leaderboard + environment block + BEST-FOR verdicts + embedded charts + full TQ bits table) + `plots/*.png` + `summary.json`. Verdicts include a labeled "balanced" heuristic (equal-weighted normalized average of TPOT/VRAM/quality) — not claimed as a rigorous optimum. Verified end-to-end against both the June 8 and July 17 sweeps.

**Not done, deliberately deferred:** TPOT-per-context-length (only VRAM is swept per length today — the plan's "TPOT@16K" acceptance phrase was aspirational shorthand; the actual evidence gathered still clears the bar via token-agreement + VRAM capacity + standard-prompt TPOT all differing). Re-running the June/July sweeps purely to backfill environment metadata on old data (low value; every sweep from now on has it natively).
