import json
import os
import argparse
import torch
from utils.config_parser import load_config
from eval.latency import VLLMLatencyProfiler
from eval.accuracy import AccuracyEvaluator
from vllm import LLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs='+', required=True, help="Path(s) to config files (e.g. model, benchmark)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to locally saved quantized model")
    parser.add_argument("--task", type=str, default="ocr")
    args = parser.parse_args()

    config = load_config(args.config)
    
    print(f"🚀 Initializing vLLM for model: {args.model_path}")
    
    # 1. Initialize vLLM Engine
    # We use a lower memory utilization for 12GB RTX 3060
    llm_profiler = VLLMLatencyProfiler(args.model_path, gpu_memory_utilization=0.7)
    
    results = {
        "model": args.model_path,
        "config": args.config,
        "metrics": {}
    }

    # 2. Latency & Throughput (using vLLM)
    print("⏲️ Running Latency & Throughput Benchmark...")
    test_prompt = "Transcribe the following text from the image accurately."
    latency_results = llm_profiler.measure_latency(test_prompt, iterations=config['benchmark'].get('timed_runs', 3))
    results["metrics"]["latency"] = latency_results

    # 3. Accuracy (using HF Datasets)
    if args.task == "ocr":
        print("🎯 Running Accuracy Evaluation (OCR)...")
        evaluator = AccuracyEvaluator(llm_profiler.llm)
        # Using CORD-v2 as a representative OCR benchmark
        results["metrics"]["accuracy"] = evaluator.eval_ocr(
            dataset_id="naver-clova-ix/cord-v2", 
            num_samples=config['benchmark'].get('eval_samples', 50)
        )

    # 4. Save Results
    os.makedirs("results", exist_ok=True)
    output_file = f"results/vllm_{os.path.basename(args.model_path)}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Evaluation complete. Results saved to {output_file}")
    print(json.dumps(results["metrics"], indent=2))

if __name__ == "__main__":
    main()
