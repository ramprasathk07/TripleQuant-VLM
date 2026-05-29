# Production Serving SLA Metrics — Study & Implementation Guide

> Purpose: learn the full set of latency/throughput SLA metrics that production LLM
> serving teams (vLLM, TensorRT-LLM, SGLang, MLPerf-Inference, OpenAI/Anthropic-scale
> serving) gate on, understand *why* each exists and how it is measured correctly,
> then implement them in this repo's benchmark harness.
>
> Scope decision (2026-05-29): build the **full async load harness**. SLA thresholds =
> **industry defaults** (TTFT ≤ 500 ms, TPOT ≤ 50 ms).
>
> **Home for all of this code: `src/evaluation/performance/`** (a dedicated package).
> Already shipped there: an **OOM-safe orchestrator** (`runner.py`) + **context-capacity
> probing** (HF real-OOM probe, vLLM KV-cache read). Everything else below lands in the
> file layout in §3. The conceptual sections (§0–§2, §4–§5) are the "study" half; §3 and
> §6 are the "where it goes / build order" half.

---

## 0. Mental model: two completely different benchmarks

The single most common benchmarking mistake is conflating these two regimes. Almost
every "our quantized model is X tok/s" number is meaningless without saying which one.

### (a) Offline / static-batch throughput
You hand the engine a fixed batch of N prompts at once and time the whole thing. This
measures **peak hardware utilization** — useful for batch/offline jobs (dataset
labeling, evals). It is NOT what an interactive product experiences.

> **This is the ONLY regime our current `measure_throughput` covers.** It replicates
> one prompt `batch_size` times and submits them in a single `generate()` call.

### (b) Online / serving (closed- vs open-loop)
Requests arrive *over time*, independently, while the engine is already busy. The
scheduler (continuous batching in vLLM, in-flight batching in TRT-LLM) interleaves
prefill and decode across requests. This is where TTFT/TPOT/goodput live, and where
quantization trade-offs actually show up (lower VRAM → bigger KV cache → more
concurrent requests → higher goodput at fixed latency).

- **Closed-loop**: a fixed number of "virtual users" each send a request, wait for the
  full response, then send the next. Measures behavior at a *concurrency* level.
- **Open-loop**: requests arrive at a target **rate** (QPS), regardless of whether
  prior ones finished — usually Poisson-distributed inter-arrival times. This is what
  exposes queueing collapse and is what vLLM's `benchmark_serving.py` and MLPerf's
  "Server" scenario use.

**We will implement open-loop (Poisson QPS) as the primary harness, with a
concurrency sweep variant.** Both are needed: open-loop for the SLA/goodput curve,
closed-loop concurrency to find the saturation point.

---

## 1. The complete metric glossary

For each: definition, formula, why it matters, and who reports it.

### 1.1 TTFT — Time To First Token
- **Def**: wall-clock from request submission to the first output token streamed back.
- **Components**: queueing delay + prefill (process all prompt tokens) + 1 decode step.
- **Why**: perceived responsiveness of streaming UIs. The headline interactive SLA.
- **Report**: mean, p50, p90, p95, p99. p99 is the SLA gate (tail latency = real UX).
- **Depends on**: prompt length (prefill is O(prompt²) attention), and queue depth
  under load. → must be measured *across context lengths* and *under load*, not once.

### 1.2 TPOT — Time Per Output Token (a.k.a. ITL mean)
- **Def**: average time between consecutive output tokens, after the first.
- **Formula**: `(E2E_latency − TTFT) / (output_tokens − 1)`.
- **Why**: streaming "smoothness" and the reciprocal of per-user generation speed
  (`tok/s/user = 1000 / TPOT_ms`). Industry chat-grade target ≈ 50 ms (20 tok/s/user).
- **Report**: mean, p50, p99.

### 1.3 ITL — Inter-Token Latency (distribution, NOT just the mean)
- **Def**: the *per-token* gap distribution, not averaged into one TPOT number.
- **Why**: TPOT mean hides jitter. A model that emits tokens in bursts (stall, then
  flush) has the same TPOT but a bad p99 ITL → visibly janky streaming. Continuous
  batching causes this: a token's latency spikes when a big new prefill is scheduled
  into its batch.
- **Report**: ITL p50/p95/p99 across all tokens of all requests.
- **Measurement requirement**: needs a true **token-streaming** API with a timestamp
  per token (see §4 — HF `TextIteratorStreamer`, vLLM streaming `RequestOutput`).

### 1.4 E2E Latency — End-to-end request latency
- **Def**: submission → last token.  `E2E = TTFT + (output_tokens − 1) * TPOT` (or just
  measured directly).
- **Why**: total time the user/agent waits. For agentic/tool-use workloads this is the
  real SLA (no streaming benefit).
- **Report**: p50/p90/p95/p99. Also **normalized E2E** = `E2E / output_tokens` (lets
  you compare across responses of different lengths).

### 1.5 Throughput — three distinct numbers (we currently report only one)
- **Output throughput**: generated tokens/sec, summed across all concurrent requests.
  The "system tok/s" headline. *(we have this)*
- **Total token throughput**: `(prompt + generated) tokens / sec`. Prefill is real
  compute; ignoring prompt tokens flatters prefill-heavy workloads. *(missing)*
- **Request throughput**: completed requests/sec (req/s). The capacity-planning
  number ($/req). *(missing)*

### 1.6 Goodput — the metric that actually gates production
- **Def**: number of requests **per second that simultaneously meet all SLAs**
  (e.g. TTFT ≤ 500 ms AND TPOT ≤ 50 ms), divided by run duration.
- **Why**: raw throughput is a lie under load — an engine can show high tok/s while
  half the requests violate TTFT. Goodput counts only "good" requests. Coined/popular-
  ized by the vLLM team; it is the right objective for serving SLAs.
- **Formula**: `goodput = #{requests where ttft ≤ T_ttft and tpot ≤ T_tpot} / duration`.
- **Report**: goodput (req/s) at each offered QPS → the **goodput curve**. The
  "max sustainable QPS" = highest offered load where goodput ≈ offered rate.

### 1.7 Prefill vs Decode throughput split
- **Def**: tokens/sec during prefill phase vs during decode phase, measured separately.
- **Why**: quantization affects them differently (weight-only quant speeds memory-bound
  decode but can slow compute-bound prefill due to dequant). Separating them explains
  *why* a scheme wins or loses. TRT-LLM and vLLM profiling both expose this.

### 1.8 Max context / KV-cache capacity
- **Def**: largest prompt the model can process on this GPU before OOM, and (serving)
  max concurrent sequences the KV cache holds.
- **Why**: directly tied to quantization value — lower weight VRAM → more KV cache →
  more concurrency.
- **Status**: ✅ DONE. HF binary-searches the real forward-pass OOM boundary; vLLM reads
  `max_model_len` + KV-cache token budget. Lives in `performance/` orchestration today.

### 1.9 Capacity / saturation
- **Def**: the offered QPS at which p99 TTFT crosses the SLA (the "knee"), and the QPS
  at which goodput stops increasing.
- **Why**: this single number is what capacity planning and autoscaling are built on.

### 1.10 Cost / efficiency (optional, nice-to-have)
- **tokens/sec/GPU**, **tokens/Joule** (via `nvidia-smi`/NVML power), **$/1M tokens**
  (tok/s × GPU $/hr). Some teams gate on $/token. Mark as phase 4 / optional.

---

## 2. Correct measurement methodology (the pitfalls)

1. **Warmup**: discard the first K requests (CUDA graph capture, lazy init, autotune).
   We already discard 5 in `measure_ttft_tpot`; keep this for the load harness too.
2. **Percentiles, not means**: SLAs are tail-based. Always p95/p99. Means hide tails.
3. **Open-loop arrivals**: use Poisson inter-arrival (`random.expovariate(qps)`) so you
   actually create queueing. A `for` loop that waits for each response is closed-loop
   and *cannot* show overload behavior.
4. **Separate TTFT from the rest via streaming**, not via two independent generate calls.
   Our current code times `generate(max_new_tokens=1)` and a separate
   `generate(max_new_tokens=64)` then subtracts — two different runs, noisy, and on HF
   it double-counts prefill. The correct way: one streamed generation, timestamp the
   first token (TTFT) and every subsequent token (ITL), derive everything from those.
5. **Fixed, realistic input/output lengths**: report them. TTFT scales with input len,
   throughput with output len. Sweep `prompt_lens` × `output_lens` (both already exist
   in `LatencyConfig` but are unused). Optionally use a length *distribution* (e.g.
   sampled from a dataset) for realism.
6. **Steady state**: measure during the middle of the run, not ramp-up/drain.
7. **Pin the clock**: `time.perf_counter()`; `torch.cuda.synchronize()` only for the
   static-batch path. For the async serving path do NOT sync between requests — that
   serializes them and destroys the point.
8. **OOM is a first-class outcome**, not a crash. Context probing and batch sweeps WILL
   OOM on small GPUs by design. Every measurement runs through `oom_safe` (already in
   `runner.py`) → returns `{"oom": True}`, frees the allocator, run continues.
9. **Report the config**: QPS, concurrency, input/output len, dtype, GPU, engine
   version. A throughput number without these is unfalsifiable.

---

## 3. Target package layout — `src/evaluation/performance/`

Everything serving-perf related lives in this one package. Status legend:
✅ done · 🔜 next · ⬜ planned.

```
src/evaluation/performance/
├── __init__.py        ✅ public API re-exports (run_perf_metrics, oom_safe, …)
├── runner.py          ✅ orchestrator: run_perf_metrics(); OOM helpers
│                          (clear_gpu_memory, is_oom_error, oom_safe)
├── records.py         🔜 RequestRecord + LoadResult dataclasses; percentile()
│                          + aggregation helpers (§7). Pure, no GPU — unit-testable.
├── streaming.py       🔜 single-request STREAMED generation per backend →
│                          {ttft_ms, itl_ms[], e2e_ms, prompt_toks, gen_toks}.
│                          stream_hf() (TextIteratorStreamer + thread),
│                          stream_vllm() (incremental RequestOutput).
├── context.py         ⬜ context-capacity: thin wrappers over runtime.measure_max_context
│                          + ctx_sweep() (TTFT vs prompt length curve). Wires the dead
│                          `ctx_sweep` / `prompt_lens` knobs.
├── load_harness.py    ⬜ async open-loop Poisson load generator (THE deliverable):
│                          run_load_test(); HF ThreadPoolExecutor driver +
│                          vLLM AsyncLLMEngine driver; concurrency/QPS sweep.
├── sla.py             ⬜ SLA tiers (table §5) + goodput math. compute_goodput(records, sla).
└── report.py          ⬜ summary extraction for _build_comparison_summary +
                           plots (goodput-vs-QPS, TTFT-vs-context, latency-throughput Pareto).
```

**Boundary rule**: the *measurement primitives* (a single streamed/static generate, the
forward-OOM probe) stay on the **runtime classes** (`HFRuntime`/`VLLMRuntime`) because
they need engine internals. The `performance/` package only **orchestrates, aggregates,
loads, scores, and reports** — it calls runtime methods, never re-implements them. This
keeps backend specifics in one place and the perf logic backend-agnostic.

### Current audit (what each existing piece does, post-refactor)

| Area | Location | State |
|---|---|---|
| Perf orchestration | `performance/runner.py::run_perf_metrics` | ✅ OOM-safe; ttft/tpot, throughput, max_context |
| OOM safety | `performance/runner.py` (`oom_safe`, `clear_gpu_memory`, `is_oom_error`) | ✅ wraps every metric + model-load in `benchmark.py` |
| `_run_perf` | `benchmark.py` | ✅ thin delegate to `run_perf_metrics` |
| TTFT/TPOT primitive (HF) | `src/runtimes/hf_runtime.py::measure_ttft_tpot` | ⚠️ two-generate subtraction; no streaming/ITL/E2E → replace via `streaming.py` |
| Throughput primitive (HF/vLLM) | `*_runtime.py::measure_throughput` | ⚠️ static single-batch; output-only tok/s |
| Max context (HF) | `hf_runtime.py::measure_max_context` | ✅ exp-grow + binary search to real OOM |
| Max context (vLLM) | `vllm_runtime.py::measure_max_context` | ✅ reads `max_model_len` + KV-cache tokens |
| Metric selection | `schemas.py::MetricsConfig.perf` | ⚠️ `ctx_sweep` literal still dead; no `e2e/itl/goodput/load_test` |
| Latency knobs | `schemas.py::LatencyConfig` | ⚠️ `prompt_lens`/`ctx_sweep` unused; no `request_rates`/`sla_*` |
| Summary | `benchmark.py::_build_comparison_summary` | ⚠️ extracts `latency` + best tok/s → extend via `report.py` |

**Verdict**: orchestration + OOM-safety + context-capacity are DONE and now packaged.
The remaining gap is the **serving layer** (streamed ITL/E2E, open-loop load, goodput,
req/s, concurrency sweep, vLLM async) → builds out the 🔜/⬜ files above.

---

## 4. Backend specifics for correct streaming measurement (goes in `streaming.py`)

You cannot get true TTFT/ITL without a per-token callback. Right API per backend.

### 4.1 Hugging Face — `stream_hf()`
- Use `transformers.TextIteratorStreamer`. Run `model.generate(..., streamer=streamer)`
  in a background thread; the main thread iterates the streamer and stamps a timestamp
  on each yielded token.
  - First yield − submit = **TTFT**; diffs between yields = **ITL**; last − submit = **E2E**.
- HF has no scheduler/continuous batching. To simulate *concurrent* load (load_harness),
  drive N requests with a `ThreadPoolExecutor` (GIL released during CUDA kernels, so
  threads overlap on GPU). Honest "naive serving" baseline — shows why you move to vLLM.

### 4.2 vLLM — `stream_vllm()`
- The current sync `LLM` class cannot do open-loop. For the load test use
  **`AsyncLLMEngine`** (`AsyncEngineArgs`) and `engine.generate(prompt, params,
  request_id)` which yields `RequestOutput` incrementally.
  - First `RequestOutput` with non-empty `outputs[0].token_ids` → TTFT; each delta → ITL.
- Launch requests as `asyncio` tasks at Poisson-scheduled times → true open-loop with
  continuous batching engaged (vLLM's actual strength).
- Mirror vLLM's `benchmarks/benchmark_serving.py` for the metric math (see §9).
- vLLM max context is already read from the engine (no probe) — done in `measure_max_context`.

---

## 5. Industry SLA thresholds (goes in `sla.py`)

Config-driven (new fields in §6 Phase 3), defaulting to the chosen interactive-chat tier:

| Tier | TTFT | TPOT (→ tok/s/user) | Use |
|---|---|---|---|
| **Industry default (chosen)** | ≤ 500 ms | ≤ 50 ms (≈20 tok/s) | general chat |
| Strict interactive | ≤ 200 ms | ≤ 25 ms (≈40 tok/s) | premium low-latency |
| Batch / throughput-first | ≤ 2000 ms | ≤ 100 ms (≈10 tok/s) | cost-optimized |

Goodput uses TTFT ≤ 500 ms AND TPOT ≤ 50 ms by default.

---

## 6. Implementation plan (phased, mapped to the §3 files)

Each phase is independently shippable and testable. Do them in order.

### Phase 0 — DONE ✅ (packaging + cheap wins already landed)
- [x] `performance/` package with `runner.py` OOM-safe orchestrator.
- [x] `oom_safe` wraps every perf metric + model-load (no run-aborting OOM).
- [x] vLLM `measure_max_context` (engine read).
- [x] `_run_perf` delegates to `run_perf_metrics`.

### Phase 0.5 — finish the cheap wins → `context.py`
- [ ] `ctx_sweep()` in `context.py`: for each len in `latency.ctx_sweep`, build a prompt
      of that token length, record TTFT (p50/p99) → `out["ctx_sweep"]`. TTFT-vs-length.
- [ ] Use `prompt_lens` × `output_lens` in the throughput path instead of one fixed prompt.
- [ ] Compose **E2E** + **total/request throughput** from data already collected.

### Phase 1 — correct single-request metrics → `streaming.py` + `records.py`
- [ ] `records.py`: `RequestRecord`, `LoadResult`, `percentile()`, aggregation (pure).
- [ ] `streaming.py`: `stream_hf()` / `stream_vllm()` → `{ttft_ms, itl_ms[], e2e_ms, …}`.
- [ ] Have `runner.run_perf_metrics` use `streaming.py` to emit `itl_ms_p*` + `e2e_ms_*`
      (kills the two-call subtraction noise). Keep old static throughput as the offline number.

### Phase 2 — async open-loop load harness → `load_harness.py` + `sla.py` (THE deliverable)
- [ ] `load_harness.run_load_test(runtime, *, qps, duration_s|num_requests, input_len,
      output_len, sla)` → `LoadResult`. Poisson arrivals (`random.expovariate(qps)`); each
      request → `RequestRecord`; aggregate TTFT/TPOT/ITL/E2E percentiles, output/total/req
      throughput, **goodput**, error rate.
- [ ] HF driver: `ThreadPoolExecutor` over `stream_hf`.
- [ ] vLLM driver: `AsyncLLMEngine` + `asyncio` task per arrival (thin adapter from same
      `EngineArgs`).
- [ ] `sla.py::compute_goodput(records, sla)`; SLA tier table (§5).
- [ ] **Concurrency / QPS sweep**: loop `request_rates` (e.g. `[1,2,4,8,16,inf]`; `inf` =
      closed-loop max) → goodput curve + saturation knee (first QPS where p99 TTFT > SLA).

### Phase 3 — schema, config, reporting → `schemas.py` + `report.py`
- [ ] `MetricsConfig.perf` Literal += `"e2e", "itl", "goodput", "load_test"`.
- [ ] `LatencyConfig` (or new `SLAConfig`): `request_rates: list[float]`,
      `sla_ttft_ms=500`, `sla_tpot_ms=50`, `load_duration_s`, `load_input_len`,
      `load_output_len`.
- [ ] `runner.run_perf_metrics`: add `e2e`/`itl`/`goodput`/`load_test` branches (all
      `oom_safe`). vLLM async path guarded; HF gets the threadpool driver.
- [ ] `report.py`: surface p99 TTFT, p99 TPOT/ITL, output tok/s, req/s, goodput@SLA,
      max-sustainable-QPS into `_build_comparison_summary`; plots (goodput-vs-QPS,
      TTFT-vs-context, latency-throughput Pareto). Tracking (wandb/mlflow) already wired.

### Phase 4 — optional efficiency → `report.py`/`runner.py`
- [ ] NVML power sampling → tokens/Joule; $/1M tokens from a configurable GPU $/hr.

---

## 7. Per-request record + aggregation schema → `performance/records.py`

```text
RequestRecord:
  arrival_s, start_s, ttft_ms, itl_ms[list], e2e_ms,
  prompt_tokens, output_tokens, success(bool), error(str|None)

LoadResult (per offered QPS):
  offered_qps, achieved_qps, duration_s, n_ok, n_err,
  ttft_ms:  {mean,p50,p90,p95,p99},
  tpot_ms:  {mean,p50,p99},      # = mean ITL per request, then percentile across reqs
  itl_ms:   {p50,p95,p99},       # pooled across all tokens
  e2e_ms:   {mean,p50,p90,p95,p99},
  throughput: {output_tok_s, total_tok_s, request_s},
  goodput_req_s, sla:{ttft_ms,tpot_ms}, error_rate
```

This nests cleanly under `results[...]["metrics"]["perf"]["load_test"] = [LoadResult,...]`
keyed by offered QPS, matching the existing crash-safe per-model JSON layout. The OOM-safe
wrapper means a `LoadResult` may instead be `{"oom": True}` for QPS levels that exhaust VRAM.

---

## 8. Validation / sanity checks before trusting numbers

- TTFT(load) ≥ TTFT(idle); TTFT p99 rises sharply past the saturation QPS (sanity: you
  actually created queueing).
- goodput ≤ offered QPS always; goodput ≈ offered QPS below the knee, then flattens.
- output_tok_s(vLLM) ≫ output_tok_s(HF) at concurrency > 1 (continuous batching works).
- ITL p99 ≥ TPOT mean (jitter exists). If ITL p99 == mean exactly, streaming timestamps
  are wrong (you're not actually streaming).
- Lower-bit quant → lower peak_vram → higher max concurrency / goodput at fixed SLA
  (this is the headline quantization story the whole repo exists to show).
- OOM markers should appear only at the high-QPS / large-context end, never at QPS=1 /
  small context (if they do, the model can't even serve one request → real problem).

---

## 9. References to study (in priority order)

1. **vLLM `benchmarks/benchmark_serving.py`** + `backend_request_func.py` — the canonical
   open-loop harness; mirror its metric math (TTFT/TPOT/ITL/E2E/goodput). Maps to
   `load_harness.py` + `records.py`.
2. **vLLM docs**: "Optimization and Tuning", "Metrics" (Prometheus: `time_to_first_token`,
   `time_per_output_token`, `e2e_request_latency`, `request_success`).
3. **The vLLM goodput / SLA-aware serving** discussion (defines goodput as SLA-attaining
   req/s). Maps to `sla.py`.
4. **MLPerf Inference** rules — "Server" scenario (Poisson arrivals, latency-bounded
   throughput) vs "Offline" scenario (static batch). Defines the regime split in §0.
5. **NVIDIA TensorRT-LLM / Triton `genai-perf`** — reports TTFT, ITL, request latency,
   throughput; good cross-check on metric definitions.
6. **DistServe / Sarathi-Serve papers** — prefill/decode disaggregation; motivates the
   prefill-vs-decode split (§1.7) and why TTFT and TPOT trade off.

---

## 10. Suggested next step (implementation order)

Phase 0 is done. Next: **Phase 0.5** (`context.py` — wire `ctx_sweep`/`prompt_lens`,
cheap, no new infra) in parallel with reading reference #1. Then **Phase 1**
(`records.py` + `streaming.py`) on HF first (easiest to verify locally on a small model);
confirm ITL p99 > TPOT mean. Then **Phase 2** (`load_harness.py` + `sla.py`): HF
threadpool driver first, then the vLLM `AsyncLLMEngine` driver. Everything stays inside
`src/evaluation/performance/`, called from `benchmark.py` via `run_perf_metrics`.
