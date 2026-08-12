# Status — personal reference

Last updated: 2026-08-12. Quick snapshot, not a narrative — see `notes/plan_v1.0.md` for
the full session-by-session log, `docs/` for the polished/public writeups.

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

| Config | TPS (HF) | TPS (vLLM) | VRAM GB | PPL Δ | MMLU Δ |
|---|---|---|---|---|---|
| FP16 | 20.9 | 56.1 | 3.28 | — | — |
| AWQ-W4A16 | 4.2 | **57.9** | 1.30 | **+15.4** | −0.032 |
| GPTQ-W8A8 | 2.8 | 49.3 | 1.95 | +0.27 | +0.008 |
| modelopt FP8 | can't execute* | blocked (WSL) | 1.95 | n/a | n/a |
| torchao int4wo | 16.7 | — | 1.47 | +14.0 | **−0.132** |
| AWQ + TurboQuant KV | 4.3 | — | 1.30 | +15.4 | −0.024 |

\* HF eager has no FP8 compute kernel — expected, not a bug.

**Headline findings:**
- Same AWQ checkpoint: 4.2 TPS on HF eager, 57.9 TPS on vLLM Marlin — **runtime kernels,
  not the quant format, decide speed.** Never judge a format's speed from eager mode.
- W4A16 hurts a lot at 1.7B scale (+15.4 PPL). W8A8 is basically free (+0.27 PPL).
- torchao int4wo is calibration-free (no AWQ-style scale search) — cheapest int4 path,
  worst quality (−0.132 MMLU, steeper than AWQ-W4A16 despite similar bit-width).

**Verification status:** every number above was cross-checked against raw JSON on
2026-08-12 (not just eyeballed). Found and fixed one real class of bug in that pass — 3
VRAM cells used GiB (÷1024) while the rest of the column used GB (÷1000), so they looked
smaller than they should relative to the other rows. Nothing else was wrong. If you spot
something that looks off in either doc, it's worth a raw-JSON check before trusting the
prose — that's exactly how the above got caught.

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
```

W&B project: `wandb.ai/New_103/triplequant-bench`. GitHub: `ramprasathk07/TripleQuant-VLM`
(tag `v1.0.0`).

---

## Next, if picking this back up

1. Decide on the WSL disk (repair vs rebuild vs leave broken) — blocks the FP8 vLLM row.
2. Maybe: 512-sample AWQ with a wider timeout / smaller model to see if the finding
   generalizes, since the full run was cost-prohibitive on Qwen3-1.7B.
3. Everything else is genuinely done — v1.0.0 is tagged, README/docs are evidence-first
   and verified, no known open correctness bugs in the shipped code path.
