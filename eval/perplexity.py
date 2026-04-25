import torch
from tqdm import tqdm

class PerplexityEvaluator:
    def __init__(self, model, tokenizer=None, processor=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer if tokenizer else processor.tokenizer
        self.device = device

    def calculate_ppl(self, dataset, max_length=2048):
        """Calculates Perplexity on a dataset."""
        self.model.eval()
        nlls = []
        
        print("Calculating Perplexity...")
        for texts in tqdm(dataset):
            # dataset should yield text strings
            encodings = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            
            input_ids = encodings.input_ids
            target_ids = input_ids.clone()
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
