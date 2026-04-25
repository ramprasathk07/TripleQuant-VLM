import argparse
import time
import torch
import os
from utils.config_parser import load_config
from data.processors import get_vlm_input
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

def load_any_model(model_path, config):
    print(f"Loading model from: {model_path}")
    trust_remote_code = config['model']['trust_remote_code']
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    
    # Simple heuristic to detect VLM vs LLM
    is_vlm = "VL" in model_path or "visual" in config.get('method', '')
    
    if is_vlm:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            device_map="auto", 
            trust_remote_code=trust_remote_code
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return model, processor, None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            device_map="auto", 
            trust_remote_code=trust_remote_code
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return model, None, tokenizer

def benchmark_model(model, processor, tokenizer, config):
    print("\n" + "="*50)
    print("RUNNING LATENCY BENCHMARK")
    print("="*50)
    
    device = next(model.parameters()).device
    prompts = config['benchmark']['sample_prompts']
    max_new_tokens = config['benchmark']['max_new_tokens']
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        # Prepare inputs
        if processor:
            inputs = get_vlm_input(processor, prompt, device=device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
        # Warmup
        for _ in range(config['benchmark']['warmup_runs']):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=5)
        
        # Timed runs
        latencies = []
        for _ in range(config['benchmark']['timed_runs']):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            latencies.append(time.time() - start)
            
        avg_latency = sum(latencies) / len(latencies)
        tokens_generated = output.shape[1] - (inputs.get('input_ids', inputs.get('pixel_values')).shape[1] if not processor else 0) # rough estimate
        
        # Decode
        if processor:
            response = processor.decode(output[0], skip_special_tokens=True)
        else:
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
        print(f"Avg Latency: {avg_latency:.3f}s")
        print(f"Sample Output: {response[:150]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, help="Override model path (useful for quantized models)")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path if args.model_path else config['model']['model_id']
    
    model, processor, tokenizer = load_any_model(model_path, config)
    benchmark_model(model, processor, tokenizer, config)
