# 2026-06-07 — TQ benchmark fixes + KV-memory & bits/accuracy metrics

Branch: main · Status: uncommitted

Session goal: make `benchmark.py -c config/benchmark/qwen3_1-7b.yaml` actually work —
fix the TurboQuant crash, fix W&B logging, explain why TQ looked inert, and add the
metrics that make TQ's benefit visible (KV memory vs context length; bits vs accuracy
vs context length).

---

## 1. TurboQuant crash in benchmark — FIXED
Root cause: the runtime restructure moved the TQ cache into `src/runtimes/hf/cache.py`
but dropped the transformers-4.55 `Cache` compat methods. `model.generate` reads
`cache.is_compileable` (base property iterates `self.layers`) and calls
`cache.get_mask_sizes(...)` (base delegates to `self.layers[idx]`) → AttributeError.
Fix: added `is_compileable = False` + `get_mask_sizes()` to `TurboQuantCache` and
`VLMCrossCache` (and `get_max_length`/`to_legacy_cache` parity on the latter).
Result: both models `status: success`, `TurboQuant enabled: 28 attn layers patched`.

## 2. W&B not logging — FIXED (two causes)
- GPU background thread logged with auto-steps, racing the explicit-step scalar logs →
  wandb dropped them → empty charts. Fix: `_init_wandb` uses `track_gpu=False`;
  `_log_record_to_wandb` logs once via `run.log(flat)` + `run.summary.update(flat)`.
- Real blocker: `.env` had a placeholder `WANDB_ENTITY=your_wandb_username_or_team_name_here`.
  `wandb.init` reads `WANDB_ENTITY` from the env directly, so `entity=None` wasn't enough —
  it failed with "entity ... not found during upsertBucket". Fix in `WandBLogger.__init__`:
  detect the placeholder, null it, AND `os.environ.pop("WANDB_ENTITY")` so wandb uses the
  API key's default account. Verified: runs log to `wandb.ai/New_103/triplequant-bench`.

## 3. Why acc/VRAM looked identical (analysis)
TurboQuant is a decode-time KV codec; it only engages in `generate()`. PPL
(`forward_logits`) and MMLU (`score_choices`) are forwards → no TQ → identical numbers.
Perf gens (≤128 tokens) were shorter than `ring_capacity=256` → ring never overflowed →
store empty → zero compression (only patched-attn overhead → TTFT 76 vs 54 ms). And
`peak_vram_mb` is dominated by the 3.4 GB weights + prefill activations, not the few-MB
KV at short lengths. So TQ was wired correctly but never exercised where it matters.

## 4. NEW metric: KV memory vs context length (`ctx_sweep`)
`HFRuntime.measure_ctx_sweep(lengths)` — prefills increasing context lengths with the
cache (TQ when enabled, else default), records **resident KV MB** (TQ-aware: ring bf16 +
`store.memory_bytes()`), peak VRAM, fits/OOM, and estimates max-context =
free_VRAM / per-token-KV-bytes. Wired into perf runner; W&B line charts
(`kv_cache_mb vs len`, `peak_vram_mb vs len`). Measured on Qwen3-1.7B:

| ctx | baseline KV | TQ KV | ratio |
|----:|------------:|------:|------:|
| 256 | 28 MB | 28 MB | 1.00x (ring holds all) |
| 1024 | 112 MB | 44 MB | 2.52x |
| 4096 | 448 MB | 110 MB | 4.07x |
| 16384 | — | 372 MB | fits (peak 8.6 GB) |

est. max context: **baseline ~67k → TQ ~348k tokens** (~5x more on the same GPU).

## 5. NEW metric: bits vs accuracy vs context length (`tq_bits_sweep`)
`HFRuntime.measure_bits_accuracy(bit_pairs, lengths, ring_capacity)` — per context length,
computes the FP16 next-token distribution, then rebuilds KV through a `TurboQuantCache` at
each `[key_bits, value_bits]` and measures, over the compressed tail `[ring:]`:
  - `top1_agreement` (argmax matches FP16)
  - `kl_div` = KL(FP16 ‖ TQ)
Real wikitext prompt; OOM-guarded. Config: `LatencyConfig.tq_bits_pairs` (default
[[2,2],[3,2],[4,4],[8,8]]), `tq_sweep_lengths`, `tq_sweep_ring`. W&B: multi-line chart
(one line per `K{kb}V{vb}`, x=context_len, y=top1_agreement) + grid table.
Status: implemented + wired; numbers from the in-flight full run pending (expect lower
bits / longer context → lower agreement; K8V8 ≈ 1.0).

## Files touched
- `src/runtimes/hf/cache.py` (Cache compat methods)
- `src/runtimes/hf/hf_runtime.py` (measure_ctx_sweep, measure_bits_accuracy, KV helpers)
- `src/evaluation/performance/runner.py` (ctx_sweep + tq_bits_sweep branches)
- `benchmark.py` (wandb: flatten, line/multi-line charts, entity-safe init)
- `src/tracking/wandb_connector.py` (placeholder-entity handling + env pop)
- `src/config/schemas.py` (perf literals: ctx_sweep already, +tq_bits_sweep; LatencyConfig
  tq_* fields)
- `config/benchmark/qwen3_1-7b.yaml` (ctx_sweep + tq_bits_sweep + lengths + tracking)

## Validation
- All files compile; config dry-run valid.
- Full benchmark run: both models success; ctx_sweep numbers above confirmed; W&B runs
  logged with metrics + ctx_sweep charts. Final run (with tq_bits_sweep) in progress.

## 6. Final verification + cleanup (2026-06-08)
- Patched-attention path validated: at exact KV (L < ring, nothing compressed) the
  patched forward == native (top1 agree 1.0, max logit diff 0.0). So bits-accuracy
  degradation is 100% quantizer fidelity, not a path bug.
- `measure_bits_accuracy` corrected to compare the LAST N positions (their recent
  context is in the exact ring — the real decode regime) instead of all `[ring:]`.
- Honest bits×accuracy on Qwen3-1.7B (top1 agreement vs FP16): K8V8 ~0.97, K4V4 ~0.3,
  K3V2 ~0.17, K2V2 ~0.13. TQ low-bit genuinely degrades this model; only K8 preserves.
- Cleanup: removed dead `current_vram_mb` (both runtimes); consolidated TQ generate
  threading into one `HFRuntime._tq_gen_kwargs(bs)` helper (was 3 inline dicts); no
  unused imports / dead module funcs in the active files.
- Final full HF run: both models success; ctx_sweep est_max 347k tokens; tq_bits_sweep
  grid + W&B multi-line charts logged. vLLM left untouched for the WSL move.

## 7. Chart/analysis report — fixes (2026-06-08)
Acting on an external chart-verifiability report:
- **Ring sawtooth bug (capture.py)** — `RingBuffer.write` flushed the ENTIRE ring on
  overflow (leaving ~1 exact token, recent context compressed). Rewrote as a true
  sliding window: keep the most-recent `capacity` exact, spill only the oldest. Real
  correctness fix for long incremental generation.
- **Disk 0.0 MB for hub models** — `_dir_size_mb(entry.path)` returned 0 for repo ids.
  Now reads the HF cache via `scan_cache_dir()`. Also fixed a Windows bug: `Path()`
  rewrites "Qwen/Qwen3-1.7B" with backslashes; compare against `str().replace("\\","/")`.
  Verified 3890 MB.
- **Throughput bs>1 misleading** — TQ is batch-1 only, so bs>1 silently ran the default
  cache. Each throughput row now carries a `turboquant` flag; chart shows which bars
  used TQ. (Verified: bs1 True/24.8 tps, bs4/8 False.)
- **top-5 agreement added** — argmax is too strict; top-5 confirms it (K3V2 top5 ~0.3–0.42
  vs top1 ~0.17). K8 top5 = 1.0.
- **Charts redesigned** — bits sweep now 3 line-series (top1/top5/KL) + grid table;
  throughput line chart; cleaner titles.
- **Typo** `RotateMatirx` → `RotateMatrix` (quantize.py).

Correction to the report's interpretation: the low bits-accuracy is NOT the sawtooth —
`measure_bits_accuracy` uses a single prefill (one `update`, last-256 already exact). The
real cause is the query attending over the compressed FAR context; K3/V2 far-key error
corrupts attention. Genuine low-bit fidelity; only K8 preserves. Ring fix still matters
for real `generate`, just not this metric.

## Deferred from the report (big / WSL)
- `logit_kl`/`token_agree` with on-disk baseline-logit caching (top5+KL added as partial).
- Memory-bandwidth (NVML/profiler) + concurrent TTFT/TPOT (async load harness; vLLM/WSL).
- Triton fused decode kernel — `score.py::_attend_compressed_only` dequantizes the whole
  cache per step, negating bandwidth savings (why TQ decode is slower). Big perf work.

## Open / next
- Confirm tq_bits_sweep grid + chart from the in-flight run.
- vLLM + TurboQuant still NOT wired (`src/turboquant_v1/vllm_backend.py` has import/name
  bugs; benchmark VLLMRuntime ignores `entry.turboquant`). Windows can't run vLLM anyway.
- Quality-under-TQ needs a generative eval (e.g. gsm8k) to route through decode; ppl/mmlu
  never will.
- Per-channel key quant (KIVI-style) is the real path to usable low-bit keys.
