import os, json, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model="{m['path']}", dtype="{m['dtype']}", gpu_memory_utilization={m['gpu_mem']},
        max_model_len={m['max_model_len']}, tensor_parallel_size={m['tp']},
        trust_remote_code=True, max_num_seqs=1)
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    def _install(worker):
        from turboquant_v1 import install_turboquant_hooks, MODE_ACTIVE
        return len(install_turboquant_hooks(worker.model_runner, key_bits=3, value_bits=2,
            buffer_size=128, mode=MODE_ACTIVE))
    hooks = executor.collective_rpc(_install)

    # Throughput
    t0 = time.perf_counter()
    out = llm.generate(["{PROMPT}"], SamplingParams(temperature=0, max_tokens=256))
    t1 = time.perf_counter()
    toks = len(out[0].outputs[0].token_ids)
    text = out[0].outputs[0].text[:200]

    # Quality (before freeing KV cache -- need paged cache for new prefill)
    def _reset(worker):
        tq_states = getattr(worker.model_runner, "_tq_states", {{}})
        for s in tq_states.values():
            s.reset()
        return len(tq_states)
    executor.collective_rpc(_reset)

    qout = llm.generate(["{QUALITY_PROMPT}"], SamplingParams(temperature=0, max_tokens=256))
    quality = qout[0].outputs[0].text[:300]

    r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_gen = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    # Free KV cache (after all generation is done)
    def _free(worker):
        from turboquant_v1 import free_kv_cache
        return free_kv_cache(worker.model_runner)
    freed = executor.collective_rpc(_free)

    r2 = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_freed = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    print(json.dumps({{"blocks": blocks, "hooks": hooks[0], "toks": toks,
        "elapsed": round(t1-t0,3), "tps": round(toks/(t1-t0),1),
        "vram_gen": vram_gen, "vram_freed": vram_freed, "freed": freed,
        "text": text, "quality": quality}}))

if __name__ == "__main__":
    main()

