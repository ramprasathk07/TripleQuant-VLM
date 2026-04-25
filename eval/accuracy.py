import torch
from datasets import load_dataset
import difflib
from vllm import LLM, SamplingParams

def calculate_wer(reference, hypothesis):
    ref_words = str(reference).lower().split()
    hyp_words = str(hypothesis).lower().split()
    if not ref_words: return 1.0
    return 1 - difflib.SequenceMatcher(None, ref_words, hyp_words).ratio()

class AccuracyEvaluator:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def eval_ocr(self, dataset_id, split="test", num_samples=50):
        """Evaluates OCR accuracy on a HuggingFace dataset."""
        print(f"Loading evaluation dataset: {dataset_id}")
        ds = load_dataset(dataset_id, split=split, streaming=True)
        
        scores = []
        sampling_params = SamplingParams(max_tokens=256, temperature=0)
        
        count = 0
        for item in ds:
            if count >= num_samples:
                break
            
            # This part needs to be adapted based on the specific dataset schema
            # Example for CORD: 'ground_truth' contains JSON, 'image' is PIL
            # For Qwen2.5-VL via vLLM, we pass the image in the prompt or multi-modal data
            
            # Note: vLLM Multi-modal support for Qwen2-VL is in recent versions
            # Format: {"prompt": "...", "multi_modal_data": {"image": image}}
            
            prompt = "Read and transcribe the text in this image."
            # Placeholder for actual multi-modal vLLM call
            # outputs = self.llm.generate([{"prompt": prompt, "multi_modal_data": {"image": item['image']}}], sampling_params)
            
            # For now, a text-based placeholder if VLM support is still being set up in vLLM
            outputs = self.llm.generate([prompt], sampling_params)
            prediction = outputs[0].outputs[0].text
            
            # Simple ground truth extraction (varies by dataset)
            ref_text = item.get('text', item.get('ground_truth', ''))
            
            acc = 1 - calculate_wer(ref_text, prediction)
            scores.append(acc)
            count += 1
            
        return {"avg_accuracy": sum(scores) / len(scores) if scores else 0}
