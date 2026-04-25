import time
import torch
import numpy as np
from vllm import LLM, SamplingParams

class VLLMLatencyProfiler:
    def __init__(self, model_path, gpu_memory_utilization=0.8):
        # Load vLLM engine
        print(f"Initializing vLLM Engine with {model_path}...")
        self.llm = LLM(
            model=model_path, 
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096 
        )

    def measure_latency(self, prompt, max_new_tokens=128, iterations=3):
        """
        Measures TTFT, TPOT, and Throughput.
        Averages over multiple iterations for stability.
        """
        ttfts = []
        tpots = [] # Time Per Output Token
        throughputs = []
        
        # Sampling params for TTFT (1 token)
        ttft_params = SamplingParams(max_tokens=1, temperature=0)
        # Sampling params for full throughput
        gen_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
        
        print(f"Benchmarking latency over {iterations} iterations...")
        
        for i in range(iterations):
            # 1. Measure TTFT
            torch.cuda.synchronize()
            t0 = time.time()
            self.llm.generate([prompt], ttft_params, use_tqdm=False)
            torch.cuda.synchronize()
            ttfts.append(time.time() - t0)
            
            # 2. Measure Throughput & TPOT
            torch.cuda.synchronize()
            t1 = time.time()
            outputs = self.llm.generate([prompt], gen_params, use_tqdm=False)
            torch.cuda.synchronize()
            t2 = time.time()
            
            total_gen_time = t2 - t1
            num_tokens = len(outputs[0].outputs[0].token_ids)
            
            if num_tokens > 1:
                # ITL/TPOT calculation (approximate since we don't have per-token timestamps here)
                # ITL = (Total Time - TTFT) / (N - 1)
                # But here we measure full gen separately to avoid prompt processing overlap twice
                tps = num_tokens / total_gen_time
                throughputs.append(tps)
                tpots.append(1.0 / tps)
        
        return {
            "avg_ttft_ms": np.mean(ttfts) * 1000,
            "avg_throughput_tps": np.mean(throughputs),
            "avg_tpot_ms": np.mean(tpots) * 1000,
            "num_tokens": max_new_tokens,
            "iterations": iterations
        }
