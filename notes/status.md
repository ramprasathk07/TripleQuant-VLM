# Status — personal reference

Last updated: 2026-08-12 (evening — MMLU significance rework). Quick snapshot, not a
narrative — see `notes/plan_v1.0.md` for the full session-by-session log, `docs/` for the
polished/public writeups.

---

## Codebase state

**Built and working:**
- 3 quantization backends: `llm_compressor` (AWQ/GPTQ/PTQ/SmoothQuant), `modelopt`
  (AWQ/PTQ, FP8/NVFP4/MXFP4 family), `torchao` (weight-only int4/int8, dynamic int8).
- 2 runtimes: HF (full metrics), vLLM (perf/serving) — dual-runtime `benchmark.py`.
- TurboQuant KV-cache compression (`src/turboquant_v1/`) — PyTorch reference only, no
  fused Triton kernels yet (5-10x slower than a fused kernel would be, scoped not built).
- Report generator (`report.py`) — JSON → report.md + charts + verdicts.
- W&B tracking, both per-model (automatic, via `benchmark.py`) and consolidated
  (`scripts/wandb_leaderboard.py` — one comparison run instead of N single-model runs).
- vLLM serving bench (`scripts/vllm_bench.py`) — runs in a separate env (WSL2, see below).

**Repo hygiene:** configs consolidated (one dir per model, underscore naming), dead
modules removed (`src/data/`, `src/integrations/`, broken `vllm_backend.py` draft).
`v1.0.0` tagged.

**Known bugs, fixed this session (2026-08-12):**
- `ModelOptQuantizer.save` was writing fake-quant weights with zero metadata (silent
  bf16, not real FP8) — now routes through `export_hf_checkpoint`. Verified: real
  `Float8_e4m3fn` tensors + `hf_quant_config.json` now present.
- torchao int4wo failed to load — default int4 kernel needs an uninstallable `mslk`
  package. Fixed by pinning `Int4PackingFormat.TILE_PACKED_TO_4D` explicitly.

**Known bugs, still open:**
- **vLLM verification of the FP8 export fix is blocked** — the WSL2 disk backing
  `/home/ramk/vllm-env` has real ext4 corruption (`dmesg`: aborted journal, filesystem
  remounted read-only). Needs `e2fsck -f` or a fresh venv in a clean distro. **Not done
  unattended** because that WSL distro also holds other in-progress TensorRT-LLM
  experimentation (from 2026-08-01, not done via Claude Code, no git trace of it) —
  repair could affect that work, so it's your call when to do it.
- AWQ-W4A16 at 512-sample calibration (does more calibration data close the +15.4 PPL
  gap at 128 samples?) — attempted, killed after ~2h when per-layer smoothing time
  scaled ~50x with 4x the samples. Genuinely unanswered.
- TurboQuant × vLLM — not integrated (draft removed in cleanup, real integration would
  be a fresh build).
- `qwen3_4b_thinking` (7 quantize configs) and `hunyuan_ocr` (5 configs, needs its own
  pinned-transformers env) — schema-valid, never actually run on GPU, ever.

---

## Results — the two things that matter

**1. TurboQuant's actual value prop (RTX 3060, 12GB):**

| Model | Max usable context |
|---|---|
| fp16 baseline | 4,096 tokens |
| torchao int8wo | 4,096 tokens |
| **TurboQuant K3V2** | **16,384 tokens — 4x** |

Cost: K3V2's next-token agreement vs fp16 is weak (9-19%) — value-bit precision is the
bottleneck, not key-bit. K8V8 is near-lossless (~97%) at only 1.3-1.7x compression.
Full chart + tradeoff table: `docs/benchmark_report.md`.

**2. Qwen3-1.7B full quantization leaderboard** (`docs/qwen3_1_7b_leaderboard.md`):

| Config | TPS (HF) | TPS (vLLM) | VRAM GB | PPL (abs) | MMLU (abs, n=800) |
|---|---|---|---|---|---|
| FP16 **(baseline)** | 21.5 | 56.1 | 3.28 | 22.45 | 0.5337 (427/800) |
| AWQ-W4A16 | 4.3 | **57.9** | 1.30 | 37.81 (**+15.4**) | 0.5225 (−1.1pp, ns) |
| GPTQ-W8A8 | 2.6 | 49.3 | 1.95 | 22.72 (+0.27) | 0.5425 (+0.9pp, ns) |
| modelopt FP8 | can't execute* | blocked (WSL) | 1.95 | n/a | n/a |
| torchao int4wo | 17.3 | — | 1.47 | 36.46 (+14.0) | 0.4350 (**−9.9pp, real**) |
| AWQ + TurboQuant KV | 4.3 | — | 1.30 | 37.85 (+15.4) | 0.5325 (−0.1pp, ns) |

\* HF eager has no FP8 compute kernel — expected, not a bug.
**FP16 metrics are not missing** — it's the baseline, so its *delta* is zero by
definition. Absolute values shown above (PPL 22.45, MMLU 0.5337). "ns" = not
statistically significant.

**Headline findings:**
- Same AWQ checkpoint: 4.3 TPS on HF eager, 57.9 TPS on vLLM Marlin — **runtime kernels,
  not the quant format, decide speed.** Never judge a format's speed from eager mode.
  This 13x gap is the motivation for `docs/kernel_learning_path.md`.
- W4A16 costs +15.4 PPL at 1.7B scale. W8A8 is basically free (+0.27 PPL).
- **Only torchao int4wo has a statistically real MMLU regression** (−9.9pp, p<0.0001).
  It's calibration-free RTN (no AWQ-style scale search) — fastest quantized row in HF
  eager *and* the worst quality.
- Mechanism worth remembering: AWQ changes 163/800 MMLU answers but symmetrically
  (86 broken / 77 fixed → net ~zero); int4wo changes 265 asymmetrically (172 / 93 → real
  damage). Same "4-bit weights" label, very different behaviour.

**Verification status:** all numbers cross-checked against raw JSON. Two real bug classes
found and fixed by doing that:
1. Three VRAM cells used GiB (÷1024) while the column used GB (÷1000).
2. **MMLU was reported without its sample size.** It was n=250, SE ±3.1pp — so the
   "quantization improved MMLU" rows were 2-3 questions flipping, and my own "AWQ costs
   −3.2pp MMLU" finding was ~1 SE of noise. Fixed at the source: `eval_mmlu_tiny` now
   returns counts + stderr + per-question outcomes, sample size is config-driven (raised
   to n=800), and `scripts/mmlu_significance.py` runs the paired McNemar exact test.
   Full writeup: `docs/failure_cases.md` #9.

Rule of thumb this produced: **if a quantized model appears to beat its own baseline,
that's the null hypothesis, not a discovery.** Quantization removes information.

---

## Quick commands

```bash
# regenerate the hero report from existing results
python report.py --dir results/qwen3-1.7b-sweep

# re-run the full quant leaderboard sweep (HF side)
python benchmark.py -c config/benchmark/qwen3_1_7b_leaderboard.yaml

# vLLM serving bench (needs a working WSL env — currently broken, see above)
wsl -- /home/ramk/vllm-env/bin/python scripts/vllm_bench.py --model <path> --quantization <fmt>

# consolidated W&B comparison view
python scripts/wandb_leaderboard.py --dir results/qwen3-1.7b-leaderboard

# is an MMLU difference real, or noise? (paired McNemar vs the fp16 baseline)
python scripts/mmlu_significance.py --dir results/qwen3-1.7b-leaderboard
```

W&B project: `wandb.ai/New_103/triplequant-bench`. GitHub: `ramprasathk07/TripleQuant-VLM`
(tag `v1.0.0`).

---

## Next, if picking this back up

1. **Kernel work** — `docs/kernel_learning_path.md` is the ramp (Level 0 concepts →
   Level 5 fused decode attention), grounded in the measured 13x eager-vs-vLLM gap.
   Existing specs it builds on: `notes/turboquant.md` §5 (Kernels A–D, pseudocode) and
   `notes/kernel_scope.md` (priorities, Triton vs TileLang). Suggested first move: the
   standalone int4 dequant benchmark (Level 2) — reproduces the 13x gap with your own
   code, cheapest path from reading about memory-bound kernels to measuring one.
2. FP8 vLLM verification — fresh `~/vllm-env2` was being built when this note was
   written; old `~/vllm-env` had ext4 journal corruption (cleared on remount, but not
   trusted). Command in the block above.
3. Open question: does 512-sample AWQ calibration recover any of the +15.4 PPL? The run
   that would answer it was killed at ~2h (per-layer smoothing scales ~50x with 4x
   samples). Would need a wider timeout or a smaller model.
4. Everything else is genuinely done — v1.0.0 tagged, docs evidence-first and verified,
   no known open correctness bugs in the shipped code path.
