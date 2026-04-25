import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from .base_quantizer import BaseQuantizer

class AWQQuantizer(BaseQuantizer):
    def quantize(self, dataset, output_dir=None):
        # Explicitly define mappings for Qwen2.5-VL to avoid automatic inference errors
        # Qwen2.5-VL-3B has 36 layers
        mappings = []
        for i in range(36):
            # Mapping 1: Attention block
            mappings.append(AWQMapping(
                smooth_layer=f"model.language_model.layers.{i}.input_layernorm",
                balance_layers=[
                    f"model.language_model.layers.{i}.self_attn.q_proj",
                    f"model.language_model.layers.{i}.self_attn.k_proj",
                    f"model.language_model.layers.{i}.self_attn.v_proj",
                ]
            ))
            # Mapping 2: MLP block
            mappings.append(AWQMapping(
                smooth_layer=f"model.language_model.layers.{i}.post_attention_layernorm",
                balance_layers=[
                    f"model.language_model.layers.{i}.mlp.gate_proj",
                    f"model.language_model.layers.{i}.mlp.up_proj",
                ]
            ))

        recipe = [
            AWQModifier(
                duo_scaling=self.config['awq'].get('duo_scaling', False),
                mappings=mappings
            ),
            QuantizationModifier(
                ignore=self.config['quantization']['ignore'],
                config_groups={
                    "group_0": {
                        "targets": self.config['quantization']['targets'],
                        "weights": {
                            "num_bits": self.config['quantization']['num_bits'],
                            "type": self.config['quantization']['weight_type'],
                            "symmetric": self.config['quantization']['symmetric'],
                            "group_size": self.config['quantization']['group_size'],
                            "strategy": self.config['quantization']['strategy'],
                            "dynamic": False,
                            "actorder": self.config['quantization'].get('actorder'),
                            "observer": self.config['quantization']['observer'],
                        },
                    }
                },
            ),
        ]

        # Use processor for VLMs, tokenizer for LLMs
        proc_or_tok = self.processor if self.processor else self.tokenizer

        oneshot(
            model=self.model,
            processor=proc_or_tok, # llmcompressor uses 'processor' for both mostly
            recipe=recipe,
            dataset=dataset,
            max_seq_length=self.config['calibration'].get('max_seq_length', 512),
            num_calibration_samples=self.config['calibration']['num_samples'],
            batch_size=self.config['calibration'].get('batch_size', 1),
            output_dir=output_dir,
            save_compressed=self.config['output'].get('save_compressed', True)
        )