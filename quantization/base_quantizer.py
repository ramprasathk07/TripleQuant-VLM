from abc import ABC, abstractmethod
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText

class BaseQuantizer(ABC):
    def __init__(self, config):
        self.config = config
        self.model_id = config['model']['model_id']
        self.torch_dtype = getattr(torch, config['model']['torch_dtype'])
        self.device_map = config['model']['device_map']
        self.trust_remote_code = config['model']['trust_remote_code']
        
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        print(f"Loading model: {self.model_id}")
        # Detect if it's a VLM or LLM
        if "VL" in self.model_id:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=self.trust_remote_code,
                min_pixels=self.config['model'].get('min_pixels'),
                max_pixels=self.config['model'].get('max_pixels')
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, 
                dtype=self.torch_dtype, 
                device_map=self.device_map, 
                trust_remote_code=self.trust_remote_code
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                dtype=self.torch_dtype, 
                device_map=self.device_map, 
                trust_remote_code=self.trust_remote_code
            )
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)

    @abstractmethod
    def quantize(self, dataset):
        pass

    def save(self, output_dir, weights=True):
        if weights:
            print(f"Saving quantized model to: {output_dir}")
            # Move to CPU to fix module_map KeyError for multi-modal models
            self.model.to("cpu")
            self.model.save_pretrained(
                output_dir, 
                save_compressed=self.config['output'].get('save_compressed', True)
            )
        if self.processor:
            self.processor.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
