from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from .base_quantizer import BaseQuantizer

class GPTQQuantizer(BaseQuantizer):
    def quantize(self, dataset):
        recipe = []
        
        if self.config.get('smoothquant', {}).get('enabled', False):
            recipe.append(SmoothQuantModifier(
                smoothing_strength=self.config['smoothquant']['smoothing_strength']
            ))
            
        recipe.append(GPTQModifier(
            targets=self.config['gptq']['targets'],
            scheme=self.config['gptq']['scheme'],
            ignore=self.config['gptq']['ignore']
        ))

        proc_or_tok = self.processor if self.processor else self.tokenizer

        oneshot(
            model=self.model,
            dataset=dataset,
            processor=self.processor if self.processor else self.tokenizer,
            recipe=recipe,
            max_seq_length=self.config['calibration']['max_seq_length'],
            num_calibration_samples=self.config['calibration']['num_samples'],
        )
